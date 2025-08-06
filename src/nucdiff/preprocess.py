import argparse
import pandas as pd
from pathlib import Path
import sys
import numpy as np
import json
import torch
from collections import Counter
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
    record_type_map = {'ADOPTED': 0, 'DECAY': 1, 'REACTION': 2, 'L': 0}
    # dataset_type 可能不存在，回退到 record_type
    if 'dataset_type' in df_featured.columns:
        base_type = df_featured['dataset_type']
    else:
        base_type = df_featured.get('record_type', pd.Series(['L']*len(df_featured)))
    df_featured['record_type_idx'] = base_type.map(record_type_map).fillna(3).astype(int)
    print("  - Detailed feature engineering complete.")
    df_featured['energy_keV'] = pd.to_numeric(df_featured['energy_keV'], errors='coerce')
    df_featured.dropna(subset=['energy_keV'], inplace=True)
    df_featured.sort_values(by=['Z_int', 'A_int', 'year', 'energy_keV'], inplace=True)

    # 路线1：为每个 (Z,A,year) 生成独立序列，并将年份作为显式token注入
    # 注：年份token形如 "<Y2006>"，Tokenizer 会在构建词表时学习到这些token。
    def create_hybrid_sequence(group):
        group = group.sort_values(by=['year', 'energy_keV'])
        group['delta_E'] = group['energy_keV'].diff().fillna(0)
        group['jp_index'] = group.groupby(['twice_J', 'parity']).cumcount()
        group['year_token'] = group['year'].apply(lambda y: f"<Y{int(y)}>")

        group['discrete_token'] = group.apply(
            lambda row: (f"{row['year_token']};"  # 年份token作为序列开头（重复于每行后续被join会保持开头位置）
                         f"J={row['twice_J']:.0f},"
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
    # 每个 (Z,A,year) 单独形成序列，以保留时间维度
    for (z, a, year), group in df_processed.groupby(['Z_int', 'A_int', 'year']):
        if len(group) > 256:
            group = group.head(256)
        # 将年份token放在整个序列的最前方（只放一次），其余行去掉重复的年份token前缀
        year_tok = f"<Y{int(year)}>"
        delta_e_sequence = group['delta_E'].tolist()
        content_tokens = group['discrete_token'].str.split(';').apply(lambda parts: ';'.join(parts[1:]))  # 去掉行级year_token
        token_sequence = ';'.join([year_tok] + content_tokens.tolist())

        final_sequences.append({
            'Z_int': int(z), 'A_int': int(a), 'year': int(year),
            'delta_e_sequence': json.dumps(delta_e_sequence),
            'token_sequence': token_sequence
        })
    df_sequences = pd.DataFrame(final_sequences)
    print(f"  - Hybrid sequences created for {len(df_sequences)} nuclides.")
    # 按 (Z,A) 聚合的全局特征仍然只与核素有关，与年份无关，按 (Z,A) 合并后会复制到每个 (Z,A,year) 序列
    nuclide_features = df_featured.groupby(['Z_int', 'A_int']).first()
    cols_to_merge = ['asym', 'is_odd_Z', 'is_odd_N', 'delta_magic_z', 'delta_magic_n', 'valence_p', 'valence_n']
    existing_cols_to_merge = [col for col in cols_to_merge if col in nuclide_features.columns]
    df_final = pd.merge(df_sequences, nuclide_features[existing_cols_to_merge], on=['Z_int', 'A_int'], how='left')
    print("  - Nuclide-level global features merged.")
    df_final_for_split = df_final.copy()
    df_final_for_split['Z'] = df_final_for_split['Z_int']
    df_final_for_split['A'] = df_final_for_split['A_int']
    train_df, val_df, _ = stratified_split_by_ZA(df_final_for_split, val_frac=val_frac, seed=seed)
    train_df.drop(columns=['Z', 'A'], inplace=True, errors='ignore')
    if not val_df.empty:
        val_df.drop(columns=['Z', 'A'], inplace=True, errors='ignore')
    print("  - Data successfully split.")
    return train_df, val_df

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
    # 列名自适配：clean_v2 的 level CSV 列包括 ['nuclide_id','record_type','raw_lines','comments','year','level_id','energy_keV','energy_unc','spin_parity','half_life_s','half_life_unc_s','is_stable','flags']
    # 需要派生 Z/A 和字符串 nuclide（如 100AG -> Z=47, A=100）
    required_cols = ['nuclide_id', 'energy_keV', 'year']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"输入CSV缺少必要列: {missing}. 实际列: {list(df.columns)}")
    
    # 解析 nuclide_id 成 (Z,A,nuclide)
    import re as _re
    def _parse_nuclide_id(nid: str):
        # 形如 '100AG' or '54FE', 即 A(>=1-3位)+元素符号
        if not isinstance(nid, str):
            return None, None, None
        m = _re.match(r'^(\d{1,3})([A-Z]{1,3})$', nid.strip())
        if not m:
            return None, None, nid
        A = int(m.group(1))
        sym = m.group(2)
        # 简易周期表映射用于 Z 推断（最常见元素，缺失则置 None）
        periodic = {
            'H':1,'HE':2,'LI':3,'BE':4,'B':5,'C':6,'N':7,'O':8,'F':9,'NE':10,
            'NA':11,'MG':12,'AL':13,'SI':14,'P':15,'S':16,'CL':17,'AR':18,'K':19,'CA':20,
            'SC':21,'TI':22,'V':23,'CR':24,'MN':25,'FE':26,'CO':27,'NI':28,'CU':29,'ZN':30,
            'GA':31,'GE':32,'AS':33,'SE':34,'BR':35,'KR':36,'RB':37,'SR':38,'Y':39,'ZR':40,
            'NB':41,'MO':42,'TC':43,'RU':44,'RH':45,'PD':46,'AG':47,'CD':48,'IN':49,'SN':50,
            'SB':51,'TE':52,'I':53,'XE':54,'CS':55,'BA':56,'LA':57,'CE':58,'PR':59,'ND':60,
            'PM':61,'SM':62,'EU':63,'GD':64,'TB':65,'DY':66,'HO':67,'ER':68,'TM':69,'YB':70,
            'LU':71,'HF':72,'TA':73,'W':74,'RE':75,'OS':76,'IR':77,'PT':78,'AU':79,'HG':80,
            'TL':81,'PB':82,'BI':83,'PO':84,'AT':85,'RN':86,'FR':87,'RA':88,'AC':89,'TH':90,
            'PA':91,'U':92,'NP':93,'PU':94,'AM':95,'CM':96,'BK':97,'CF':98,'ES':99,'FM':100
        }
        Z = periodic.get(sym, None)
        nuclide = f"{A}{sym}"
        return Z, A, nuclide
    
    parsed = df['nuclide_id'].apply(_parse_nuclide_id)
    df[['Z','A','nuclide']] = pd.DataFrame(parsed.tolist(), index=df.index)
    
    # 统一列名并清洗
    df.rename(columns={
        'flags': 'flag',
        'half_life_s': 'half_life',  # 与后续解析接口统一
    }, inplace=True)
    
    df.dropna(subset=['nuclide', 'energy_keV', 'Z', 'A', 'year'], inplace=True)
    df[['Z', 'A', 'year']] = df[['Z', 'A', 'year']].astype(int)
    print(f"加载完成。初始共 {len(df)} 条能级记录。")
    
    # 保留所有年份记录；仅同一年轻量去重
    df['flag_priority'] = df.get('flag', pd.Series(index=df.index, dtype=object)).apply(lambda x: 0 if x == 'A' else 1)
    df.sort_values(by=['Z', 'A', 'year', 'flag_priority'], inplace=True)
    duplicate_cols = ['Z', 'A', 'year', 'energy_keV', 'spin_parity']
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in duplicate_cols if c in df.columns], keep='first')
    after = len(df)
    df.drop(columns='flag_priority', inplace=True, errors='ignore')
    print(f"同年轻量去重：移除 {before - after} 条完全重复，跨年全部保留。")
    
    val_frac = cfg.get("val_frac", 0.2)
    seed = cfg.get("seed", 42)

    train_df, val_df = process_for_sequence_generation(df, val_frac, seed)
    
    print("\n--- 最终数据预览 ---")
    print("Train DF Head:")
    print(train_df.head())
    print("\nVal DF Head:")
    print(val_df.head())
    
    matches = re.findall(r'\d{4}', Path(args.input_file).stem)
    start_year, end_year = (0,0)
    if matches:
        years = sorted([int(y) for y in matches])
        start_year, end_year = years[0], years[-1]
    
    path_cfg = { "seed": seed, "val_frac": val_frac, "base_start_year": start_year, "base_end_year": end_year }
    train_path, val_path = get_preprocessed_paths(path_cfg, "SEQ")
    train_path.parent.mkdir(exist_ok=True, parents=True)
    
    final_cols = sorted(train_df.columns.tolist())
    train_df[final_cols].to_feather(train_path)
    if not val_df.empty:
        val_df[final_cols].to_feather(val_path)

    # 额外导出 Route-2 时间增广的索引元数据：每个 (Z,A) 对应有哪些年份的样本行索引
    # 以便训练时在固定 seed 和 epoch 下做可复现采样（k-year per epoch）
    # 结构示例：
    # {
    #   "map": { "Z_A": { "years": [2005,2006,...], "idx": { "2005": [row_ids...], "2006": [row_ids...] } }, ... },
    #   "k_default": 2
    # }
    try:
        # 只基于训练集构建映射，避免验证集泄漏
        map_obj = {}
        for (z, a), grp in train_df.groupby(['Z_int', 'A_int']):
            key = f"{int(z)}_{int(a)}"
            years = sorted(grp['year'].astype(int).unique().tolist())
            idx_map = {}
            for y in years:
                # 记录在训练集 DataFrame 内的行位置索引（0..len(train_df)-1）
                pos_idx = grp.index[grp['year'] == y].tolist()
                # Feather 保存后索引会重置；因此我们转换为“行号”序列（相对于 train_df 的位置）
                # 使用 enumerate over train_df.index 来建立 index->position 的映射
                # 为避免 O(N) 建表两次，直接用 RangeIndex 假设（训练集保存前是默认 RangeIndex）
                # 若不是 RangeIndex，则回退到位置定位：
                if not isinstance(train_df.index, pd.RangeIndex):
                    # 通过 get_indexer 获取位置
                    pos = train_df.index.get_indexer(pos_idx)
                    idx_map[str(y)] = [int(p) for p in pos if p >= 0]
                else:
                    idx_map[str(y)] = [int(i) for i in pos_idx]
            map_obj[key] = {"years": years, "idx": idx_map}
        year_index_meta = {
            "map": map_obj,
            "k_default": 2
        }
        with open(train_path.parent / "year_index_map.json", "w", encoding="utf-8") as f:
            json.dump(year_index_meta, f, indent=2, ensure_ascii=False)
        print(f"  - 已导出年份索引映射: {str((train_path.parent / 'year_index_map.json').resolve())}")
    except Exception as e:
        print(f"  - 警告：导出年份索引映射失败：{e}")
    
    print("\n--- 构建并保存分词器和类别权重 ---")
    tokenizer = LevelTokenizer()
    # 预注册年份token以避免被当作UNK；也支持动态年份范围
    years_in_data = sorted(set(train_df['year'].astype(int).tolist()))
    for y in years_in_data:
        tok = f"<Y{y}>"
        if tok not in tokenizer.token_to_id:
            idx = len(tokenizer.token_to_id)
            tokenizer.token_to_id[tok] = idx
            tokenizer.id_to_token[idx] = tok
    # 然后基于序列构建剩余词表
    tokenizer.build_vocab(train_df['token_sequence'])
    tokenizer_path = train_path.parent / "tokenizer_vocab.json"
    tokenizer.save(tokenizer_path)
    print(f"  - 分词器词汇表已保存到: {tokenizer_path}")
    
    all_tokens = [tok for seq in train_df['token_sequence'] for tok in seq.split(';')]
    counts = Counter(all_tokens)
    # Use sqrt-based smoothing to avoid extreme weights (instead of 1/count)
    total_count = len(all_tokens)
    weights = {tok: (total_count / count) ** 0.5 for tok, count in counts.items()}
    # Normalize to have mean = 1.0
    mean_weight = np.mean(list(weights.values()))
    weights = {tok: w / mean_weight for tok, w in weights.items()}
    
    class_weights = torch.ones(len(tokenizer))
    for token, weight in weights.items():
        if token in tokenizer.token_to_id:
            class_weights[tokenizer.token_to_id[token]] = weight
    print(f"  - 已计算 {len(class_weights)} 个词元的类别权重。")
    
    # NOTE: delta_e_scaler is now deprecated in favor of normalization in the training script
    metadata = {
        'description': 'Preprocessed data for hybrid seq2seq model.',
        'final_columns': final_cols,
        'class_weights': class_weights.tolist()
    }
    with open(train_path.parent / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"\n  - 数据和元数据（包含类别权重）已成功保存到: {train_path.parent.resolve()}")
    print(f"\n--- 预处理完成 ---")

if __name__ == "__main__":
    main()
