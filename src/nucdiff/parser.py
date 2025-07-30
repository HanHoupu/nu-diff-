import re
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, asdict, field
import itertools
from typing import Optional, Tuple

from src.nucdiff.utils.constants import ELEMENTS

# ID generator
level_id_counter = itertools.count()

@dataclass
class Level:
    nuclide: str
    Z: int
    A: int
    energy_keV: float
    spin_parity: str
    half_life: str
    source_dataset_id: str
    dataset_type: str = '' # e.g., 'ADOPTED', 'DECAY', 'REACTION'
    flag: str = '' # e.g., 'A', 'B', 'S'
    is_stable: bool = False
    level_id: int = field(init=False)

    def __post_init__(self):
        self.level_id = next(level_id_counter)

@dataclass
class Gamma:
    """Represents a single gamma transition."""
    level_id: int
    nuclide: str
    Z: int
    A: int
    gamma_energy_keV: float
    gamma_intensity: float
    gamma_decay_mult: str
    gamma_mixing_ratio: float

_float_pat = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?")

@dataclass
class ParseStats:
    lines_processed: int = 0
    datasets_found: int = 0
    levels_found: int = 0
    gammas_found: int = 0
    skipped_lines_other_rtype: int = 0
    skipped_lines_invalid_level: int = 0
    skipped_lines_invalid_gamma: int = 0

    def report(self):
        print("\n--- 解析过程统计报告 ---")
        print(f"总处理行数: {self.lines_processed}")
        print(f"识别出的数据集数量: {self.datasets_found}")
        print(f"成功解析的能级记录: {self.levels_found}")
        print(f"成功解析的伽马记录: {self.gammas_found}")
        print(f"因非 L/G 类型而跳过的行数: {self.skipped_lines_other_rtype}")
        print(f"因无效能级数据而跳过的行数: {self.skipped_lines_invalid_level}")
        print(f"因无效伽马数据而跳过的行数: {self.skipped_lines_invalid_gamma}")
        print("--------------------------\n")

class ENSDFParserV2:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _parse_float(self, s: str) -> float:
        match = _float_pat.search(s.replace(" ", ""))
        if match:
            try:
                return float(match.group(0))
            except (ValueError, OverflowError):
                return np.nan
        return np.nan

    def _parse_nuclide_from_token(self, token: str) -> tuple[str, int, int]:
        """从一个字符串token (例如 '3H', '110AG') 中解析出 nuclide, A, 和 Z。"""
        match = re.match(r"(\d+)([A-Za-z]+)", token)
        if not match:
            return "UNKNOWN", 0, 0
        
        mass_str, sym_str = match.groups()
        sym_str = sym_str.upper()
        a = int(mass_str)
        z = ELEMENTS.get(sym_str, 0)

        if z == 0:
            return "UNKNOWN", 0, 0
        
        return sym_str, a, z

    def _parse_nuclide_str(self, nuc_str: str) -> tuple[str, int, int]:
        # Flawed but working version
        mass_str = nuc_str[:3].strip()
        sym_str = nuc_str[3:5].strip().upper()
        if not mass_str.isdigit() or not sym_str:
            return "UNKNOWN", 0, 0
        a = int(mass_str)
        z = ELEMENTS.get(sym_str, 0)
        if z == 0:
            return "UNKNOWN", 0, 0
        return sym_str, a, z

    def _split_datasets(self, lines: list[str]) -> list[list[str]]:
        """
        根据前5列核心核素ID (例如 '3H', '4LI') 的变化来分割数据集。
        这种方法比之前比较原始字符串的方式更健壮。
        """
        datasets = []
        current_dataset_lines = []
        last_core_nucid = None

        for line in lines:
            if not line.strip() or len(line) < 5:
                continue
            
            # 从行首提取核心ID
            current_core_nucid = self._get_core_nucid(line[:5])

            # 如果一个有效的ID出现了，并且和上一个不同，则开启一个新数据集
            if current_core_nucid and current_core_nucid != last_core_nucid:
                if current_dataset_lines:
                    datasets.append(current_dataset_lines)
                current_dataset_lines = [line]
                last_core_nucid = current_core_nucid
            else: # 否则，继续添加到当前数据集中
                current_dataset_lines.append(line)
        
        # 不要忘记最后一个数据集
        if current_dataset_lines:
            datasets.append(current_dataset_lines)
            
        return datasets

    def _parse_dataset(self, lines: list[str], stats: ParseStats) -> tuple[list[Level], list[Gamma]]:
        """
        Parses a single dataset, now using token-based identification and
        fixed-width extraction for robustness.
        """
        levels = []
        gammas = []
        current_level: Level | None = None
        
        if not lines:
            return [], []

        source_id_str = lines[0][:5].strip()
        
        # 从数据集标题确定数据集类型
        dataset_type = "UNKNOWN"
        if lines:
            title_line = lines[0].upper()
            if "ADOPTED LEVELS" in title_line:
                dataset_type = "ADOPTED"
            elif "DECAY" in title_line:
                dataset_type = "DECAY"
            elif "REACTION" in title_line or "(" in title_line:
                dataset_type = "REACTION"

        for line in lines:
            words = line.split()
            if len(words) < 2:
                continue

            # 使用用户建议的、更健壮的token-based方法来识别记录类型
            rtype = words[1].upper()

            if rtype not in ['L', 'G']:
                stats.skipped_lines_other_rtype += 1
                continue
            
            # 识别成功后，使用固定宽度切片来精确提取数据
            if rtype == 'L':
                # 对核素的解析也使用更健壮的方法
                sym, A, Z = self._parse_nuclide_from_token(words[0])
                if Z == 0:
                    stats.skipped_lines_invalid_level += 1
                    continue
                
                # 确保行足够长以进行切片
                if len(line) < 80:
                    stats.skipped_lines_invalid_level += 1
                    continue

                try:
                    energy_str = line[9:19].strip()
                    if not energy_str or '?' in energy_str or any(c.isalpha() for c in energy_str.replace('E','').replace('e','').replace('.','').replace('+','-')):
                        stats.skipped_lines_invalid_level += 1
                        continue
                    
                    energy_val = self._parse_energy(energy_str)
                    if energy_val is None:
                        stats.skipped_lines_invalid_level += 1
                        continue

                except (ValueError, IndexError):
                    stats.skipped_lines_invalid_level += 1

                spin_parity_val = line[21:39].strip()
                half_life_val = line[39:49].strip()
                flag_val = line[54] if len(line) > 54 else ''

                # 恢复自旋宇称的格式验证，以过滤掉格式错误的注释行等
                spin_parity_re = re.compile(r'^[0-9/().+ -]*$')
                if spin_parity_val and not spin_parity_re.match(spin_parity_val):
                    stats.skipped_lines_invalid_level += 1
                    continue

                is_stable_flag = 'STABLE' in half_life_val.upper()

                current_level = Level(
                    nuclide=f"{sym}-{A}",
                    Z=Z,
                    A=A,
                    energy_keV=energy_val,
                    spin_parity=spin_parity_val,
                    half_life=half_life_val,
                    source_dataset_id=source_id_str,
                    dataset_type=dataset_type,
                    flag=flag_val,
                    is_stable=is_stable_flag,
                )
                levels.append(current_level)
                stats.levels_found += 1

            elif rtype == 'G' and current_level:
                if len(line) < 50: # 确保行足够长
                    stats.skipped_lines_invalid_gamma += 1
                    continue

                energy_val = self._parse_float(line[9:19])
                intensity_val = self._parse_float(line[21:29])
                if np.isnan(energy_val):
                    stats.skipped_lines_invalid_gamma += 1
                    continue

                gamma = Gamma(
                    level_id=current_level.level_id,
                    nuclide=current_level.nuclide,
                    Z=current_level.Z,
                    A=current_level.A,
                    gamma_energy_keV=energy_val,
                    gamma_intensity=intensity_val,
                    gamma_decay_mult=line[31:39].strip(),
                    gamma_mixing_ratio=self._parse_float(line[40:48])
                )
                gammas.append(gamma)
                stats.gammas_found += 1
        return levels, gammas

    def _parse_energy(self, energy_str: str) -> Optional[float]:
        """
        Robustly parses an energy string, handling scientific notation and potential junk.
        """
        # Regular expression to find the first valid floating-point number
        # This handles cases like "20.21 2", "1.234E+03", "5000. 10"
        match = re.search(r'[+-]?\d+\.?\d*(?:[Ee][+-]?\d+)?', energy_str)
        if match:
            try:
                energy_val = float(match.group(0))
                # Automatic keV conversion for values that are clearly in MeV
                # e.g., energies between 3000 and 100000 are likely MeV, not keV.
                if 3000 < energy_val < 100000:
                    energy_val /= 1000.0
                return energy_val
            except ValueError:
                return None
        return None

    def _get_core_nucid(self, line: str) -> Optional[Tuple[str, str]]:
        """
       从前5列中提取核心核素ID，例如 '  3H ' -> '3H'
        """
        if len(line) < 5:
            return None
        mass_str = line[:3].strip()
        sym_str = line[3:5].strip()
        if mass_str.isdigit() and sym_str.isalpha():
            return f"{mass_str}{sym_str}"
        return None

    def parse_file(self, file_path_base: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parses all relevant ENSDF files for a given base path (e.g., "ENSDF-2005").
        It automatically finds and reads .L, .G, and .Q files.
        """
        print(f"正在从基础路径 {Path(file_path_base).name} 解析数据...")
        base = Path(file_path_base)
        lines = []
        # Look for all possible extensions and concatenate their content
        for ext in ['', '.L', '.G', '.Q']:
            filepath = base.with_suffix(ext)
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        lines.extend(f.readlines())
                    if self.verbose:
                        print(f"  - 已加载 {filepath.name}")
                except Exception as e:
                    print(f"Warning: Could not read file {filepath}: {e}")
        
        if not lines:
            print(f"Warning: No data lines found for base path {base.name}")
            return self.build_dataframes([], [])
        
        stats = ParseStats()
        stats.lines_processed = len(lines)
        datasets = self._split_datasets(lines)
        stats.datasets_found = len(datasets)

        all_levels, all_gammas = [], []
        for ds_lines in tqdm(datasets, desc="正在处理数据集", disable=not self.verbose):
            levels, gammas = self._parse_dataset(ds_lines, stats)
            all_levels.extend(levels)
            all_gammas.extend(gammas)
        
        stats.report()
        return self.build_dataframes(all_levels, all_gammas)

    def build_dataframes(self, levels: list[Level], gammas: list[Gamma]) -> tuple[pd.DataFrame, pd.DataFrame]:
        print("正在将解析结果构建为DataFrame...")
        levels_df = pd.DataFrame([asdict(lv) for lv in levels])
        gammas_df = pd.DataFrame([asdict(g) for g in gammas])
        print(f"成功构建DataFrame: {len(levels_df)}个能级, {len(gammas_df)}个伽马。")
        return levels_df, gammas_df

    def export_to_csv(self, levels_df: pd.DataFrame, gammas_df: pd.DataFrame, output_dir: str, year_tag: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        levels_path = output_path / f"levels_{year_tag}.csv"
        gammas_path = output_path / f"gammas_{year_tag}.csv"
        print(f"正在导出数据到 {output_path.resolve()}...")
        levels_df.to_csv(levels_path, index=False)
        gammas_df.to_csv(gammas_path, index=False)
        print(f"数据导出完成: {levels_path.name}, {gammas_path.name}")
