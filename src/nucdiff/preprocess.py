import argparse
import pandas as pd
from pathlib import Path
import sys
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
import re
import yaml # Added for loading config file

# Physical constants and conversion factors
HBAR_MEV_S = 6.582119569e-22  # MeV*s
LN2 = np.log(2)

# Conversion factors for half-life/width strings to lifetime in seconds
# For time units (Y, D, ...): factor to convert value to seconds
# For energy units (MEV, KEV, ...): factor to convert 1/value to seconds (ħ/Γ)
TIME_UNITS_TO_TAU_S = {
    'Y': 3.1536e7 / LN2, 'D': 86400 / LN2, 'H': 3600 / LN2, 'MIN': 60 / LN2, 'S': 1.0 / LN2,
    'MS': 1e-3 / LN2, 'US': 1e-6 / LN2, 'NS': 1e-9 / LN2, 'PS': 1e-12 / LN2, 'FS': 1e-15 / LN2, 'AS': 1e-18 / LN2,
    'KEV': HBAR_MEV_S / 1e-3, 'EV': HBAR_MEV_S / 1e-6, 'MEV': HBAR_MEV_S
}

def parse_lifetime(hl_str: str) -> tuple[float, int, int]:
    if pd.isna(hl_str) or not isinstance(hl_str, str) or hl_str.strip() == '':
        return -25.0, 0, 0
    s = hl_str.upper().strip()
    if s == 'STABLE':
        return 20.0, 1, 0
    is_censored = 1 if s.startswith(('>', '<', '~', 'AP')) else 0
    s = re.sub(r'^[><~AP]+', '', s).strip()
    match = re.match(r'([0-9.E+-]+)\s*([A-Z]+)', s)
    if not match:
        return -25.0, 0, 0
    value_str, unit = match.groups()
    try:
        value = float(value_str)
        if value <= 0:
            return -25.0, 0, 0
    except ValueError:
        return -25.0, 0, 0
    if unit in TIME_UNITS_TO_TAU_S:
        if unit in ('Y', 'D', 'H', 'MIN', 'S', 'MS', 'US', 'NS', 'PS', 'FS', 'AS'):
            tau_s = value * TIME_UNITS_TO_TAU_S[unit]
        else:
            tau_s = TIME_UNITS_TO_TAU_S[unit] / value
    else:
        return -25.0, 0, 0
    log_tau = np.log10(tau_s)
    log_tau_clipped = np.clip(log_tau, -25.0, 25.0)
    return log_tau_clipped, 1, is_censored

def parse_spin_parity(sp_str: str) -> tuple[int, int]:
    if pd.isna(sp_str) or not isinstance(sp_str, str) or sp_str.strip() == '':
        return -1, 0
    s = sp_str.strip()
    parity_val = 0
    if s.endswith('+'):
        parity_val = 1
        s = s[:-1]
    elif s.endswith('-'):
        parity_val = -1
        s = s[:-1]
    s = s.strip('()')
    if ',' in s:
        s = s.split(',')[0].strip()
    try:
        if '/' in s:
            num, den = map(int, s.split('/'))
            if den == 2:
                return num, parity_val
            else:
                return int(2 * float(num) / float(den)), parity_val
        else:
            return int(s) * 2, parity_val
    except (ValueError, TypeError):
        return -1, 0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.nucdiff.utils.paths import get_preprocessed_paths
from src.nucdiff.utils.constants import STABLE_NUCLIDES_BACKBONE
from src.nucdiff.utils.features import add_engineered_features
from src.nucdiff.data.tokenizer import LevelTokenizer

def stratified_split_by_ZA(df: pd.DataFrame, val_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, set]:
    np.random.seed(seed)
    # Ensure Z and A columns exist before creating nuclide_id
    if 'Z' not in df.columns or 'A' not in df.columns:
        raise ValueError("DataFrame must contain 'Z' and 'A' columns for splitting.")
    df['nuclide_id'] = list(zip(df['Z'], df['A']))
    all_nuclides = df['nuclide_id'].unique()
    backbone_nuclides = set((z, a) for z, a_list in STABLE_NUCLIDES_BACKBONE.items() for a in a_list)
    data_nuclides_set = set(map(tuple, all_nuclides))
    actual_backbone_in_data = data_nuclides_set.intersection(backbone_nuclides)
    all_elements_in_data = df['Z'].unique()
    elements_with_preset_backbone = {z for z, a in actual_backbone_in_data}
    elements_without_backbone = set(all_elements_in_data) - elements_with_preset_backbone
    if elements_without_backbone:
        for z in sorted(list(elements_without_backbone)):
            element_df = df[df['Z'] == z]
            if not element_df.empty:
                most_common_A = element_df['A'].value_counts().idxmax()
                new_backbone_nuclide = (z, int(most_common_A))
                actual_backbone_in_data.add(new_backbone_nuclide)
    final_backbone_nuclides = actual_backbone_in_data
    leaf_nuclides = data_nuclides_set - final_backbone_nuclides
    leaf_nuclides_list = sorted(list(leaf_nuclides))
    val_count = 1 if len(leaf_nuclides_list) < 5 else max(1, round(len(leaf_nuclides_list) * val_frac))
    if not leaf_nuclides_list: val_count = 0
    shuffled_leaf_nuclides = np.random.permutation(leaf_nuclides_list)
    val_leaf_nuclides = set(map(tuple, shuffled_leaf_nuclides[:val_count]))
    train_leaf_nuclides = set(map(tuple, shuffled_leaf_nuclides[val_count:]))
    final_train_nuclides = final_backbone_nuclides.union(train_leaf_nuclides)
    final_val_nuclides = val_leaf_nuclides
    train_df = df[df['nuclide_id'].isin(list(final_train_nuclides))].copy()
    val_df = df[df['nuclide_id'].isin(list(final_val_nuclides))].copy()
    train_df.drop(columns=['nuclide_id'], inplace=True, errors='ignore')
    if not val_df.empty:
        val_df.drop(columns=['nuclide_id'], inplace=True, errors='ignore')
    return pd.DataFrame(train_df), pd.DataFrame(val_df), final_val_nuclides

def process_for_sequence_generation(df: pd.DataFrame, val_frac: float, seed: int):
    print("\n--- [Sequence Generation] Starting new preprocessing pipeline ---")
    df_featured = add_engineered_features(df.copy())
    df_featured.rename(columns={'Z': 'Z_int', 'A': 'A_int'}, inplace=True)
    spin_data = df_featured['spin_parity'].apply(parse_spin_parity)
    df_featured[['twice_J', 'parity']] = pd.DataFrame(spin_data.tolist(), index=df_featured.index)
    df_featured['twice_J'] += 1
    df_featured['parity'] += 1
    lifetime_data = df_featured['half_life'].apply(parse_lifetime)
    lifetime_cols = ['log10_tau_s', 'has_tau', 'is_lifetime_censored']
    df_featured[lifetime_cols] = pd.DataFrame(lifetime_data.tolist(), index=df_featured.index)
    df_featured['is_isomer'] = ((df_featured['energy_keV'] > 0) & (df_featured['has_tau'] == 1) & (df_featured['log10_tau_s'] > -6)).astype(int)
    record_type_map = {'ADOPTED': 0, 'DECAY': 1, 'REACTION': 2}
    df_featured['record_type_idx'] = df_featured['dataset_type'].map(record_type_map).fillna(3).astype(int)
    print("  - Detailed feature engineering complete.")
    df_featured['energy_keV'] = pd.to_numeric(df_featured['energy_keV'], errors='coerce')
    df_featured.dropna(subset=['energy_keV'], inplace=True)
    df_featured.sort_values(by=['Z_int', 'A_int', 'year', 'energy_keV'], inplace=True)
    def create_hybrid_sequence(group):
        group['delta_E'] = group['energy_keV'].diff().fillna(0)
        group['jp_index'] = group.groupby(['twice_J', 'parity']).cumcount()
        group['discrete_token'] = group.apply(
            lambda row: (f"J={row['twice_J']:.0f},"
                         f"P={row['parity']:.0f},"
                         f"T={row['log10_tau_s']:.2f},"
                         f"I={row['is_isomer']},"
                         f"R={row['record_type_idx']},"
                         f"N={row['jp_index']:.0f}"),
            axis=1
        )
        return group
    df_processed = df_featured.groupby(['Z_int', 'A_int'], group_keys=False).apply(create_hybrid_sequence)
    final_sequences = []
    for name, group in df_processed.groupby(['Z_int', 'A_int']):
        if len(group) > 256:
            group = group.head(256)
        delta_e_sequence = group['delta_E'].tolist()
        token_sequence = ';'.join(group['discrete_token'])
        latest_year = group['year'].max()
        final_sequences.append({
            'Z_int': name[0], 'A_int': name[1], 'year': latest_year,
            'delta_e_sequence': json.dumps(delta_e_sequence),
            'token_sequence': token_sequence
        })
    df_sequences = pd.DataFrame(final_sequences)
    print(f"  - Hybrid sequences created for {len(df_sequences)} nuclides.")
    nuclide_features = df_featured.groupby(['Z_int', 'A_int']).first()
    cols_to_merge = ['asym', 'is_odd_Z', 'is_odd_N', 'delta_magic_z', 'delta_magic_n', 'valence_p', 'valence_n']
    existing_cols_to_merge = [col for col in cols_to_merge if col in nuclide_features.columns]
    df_final = pd.merge(df_sequences, nuclide_features[existing_cols_to_merge], on=['Z_int', 'A_int'], how='left')
    print("  - Nuclide-level global features merged.")
    df_final_for_split = df_final.copy()
    df_final_for_split['Z'] = df_final_for_split['Z_int']
    df_final_for_split['A'] = df_final_for_split['A_int']
    train_df_for_scaler, _, _ = stratified_split_by_ZA(df_final_for_split, val_frac=val_frac, seed=seed)
    all_delta_e_train = [d for sublist in train_df_for_scaler['delta_e_sequence'].apply(json.loads) for d in sublist]
    delta_e_mean = np.mean(all_delta_e_train) if all_delta_e_train else 0.0
    delta_e_std = np.std(all_delta_e_train) if all_delta_e_train else 1.0
    print(f"  - Calculated delta_E scaler from training set: mean={delta_e_mean:.4f}, std={delta_e_std:.4f}")
    train_df, val_df, _ = stratified_split_by_ZA(df_final_for_split, val_frac=val_frac, seed=seed)
    train_df.drop(columns=['Z', 'A'], inplace=True, errors='ignore')
    if not val_df.empty:
        val_df.drop(columns=['Z', 'A'], inplace=True, errors='ignore')
    print("  - Data successfully split.")
    return train_df, val_df, (delta_e_mean, delta_e_std)

def main():
    parser = argparse.ArgumentParser(description="从清洗后的CSV文件创建特征、分割并保存最终用于训练的数据集。")
    parser.add_argument("--input_file", type=str, required=True, help="由clean.py生成的清洗后的levels.csv文件路径。")
    parser.add_argument("--cfg", type=str, default=None, help="Path to the configuration file to ensure consistent settings.")
    parser.add_argument("--task", type=str, default="SEQ", choices=["SEQ"], help="任务类型 (当前仅支持 'SEQ' 用于序列模型)。")
    args = parser.parse_args()
    cfg = {}
    if args.cfg:
        with open(args.cfg, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    print(f"\n[步骤 1/4] 从 {args.input_file} 加载已清洗的数据...")
    df = pd.read_csv(args.input_file)
    df.dropna(subset=['nuclide', 'energy_keV', 'Z', 'A', 'year'], inplace=True)
    df[['Z', 'A', 'year']] = df[['Z', 'A', 'year']].astype(int)
    print(f"加载完成。初始共 {len(df)} 条能级记录。")
    df['flag_priority'] = df['flag'].apply(lambda x: 0 if x == 'A' else 1)
    df.sort_values(by='flag_priority', inplace=True)
    duplicate_cols = ['Z', 'A', 'energy_keV', 'spin_parity']
    df.drop_duplicates(subset=duplicate_cols, keep='first', inplace=True)
    df.drop(columns='flag_priority', inplace=True)
    print(f"去重后剩余 {len(df)} 条能级记录。")
    
    # Use config for val_frac and seed
    val_frac = cfg.get("val_frac", 0.2)
    seed = cfg.get("seed", 42)

    train_df, val_df, (delta_e_mean, delta_e_std) = process_for_sequence_generation(df, val_frac, seed)
    
    print("\n--- 最终数据预览 ---")
    print("Train DF Head:")
    print(train_df.head())
    print("\nVal DF Head:")
    print(val_df.head())
    
    # Extract year range from the filename for path generation
    matches = re.findall(r'\d{4}', Path(args.input_file).stem)
    start_year, end_year = (0,0)
    if matches:
        years = sorted([int(y) for y in matches])
        start_year, end_year = years[0], years[-1]
    
    path_cfg = {
        "seed": seed, "val_frac": val_frac,
        "base_start_year": start_year, "base_end_year": end_year
    }
    train_path, val_path = get_preprocessed_paths(path_cfg, "SEQ")
    train_path.parent.mkdir(exist_ok=True, parents=True)
    
    final_cols = sorted(train_df.columns.tolist())
    train_df[final_cols].to_feather(train_path)
    if not val_df.empty:
        val_df[final_cols].to_feather(val_path)
    
    print("\n--- 构建并保存分词器 ---")
    tokenizer = LevelTokenizer()
    tokenizer.build_vocab(train_df['token_sequence'])
    tokenizer_path = train_path.parent / "tokenizer_vocab.json"
    tokenizer.save(tokenizer_path)
    print(f"  - 分词器词汇表已保存到: {tokenizer_path}")
    
    metadata = {
        'description': 'Preprocessed data for hybrid seq2seq model.',
        'final_columns': final_cols,
        'delta_e_scaler': {'mean': delta_e_mean, 'std': delta_e_std}
    }
    with open(train_path.parent / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"\n  - 数据和元数据已成功保存到: {train_path.parent.resolve()}")
    print(f"\n--- 预处理完成 ---")

if __name__ == "__main__":
    main()
