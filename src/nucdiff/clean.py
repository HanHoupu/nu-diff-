#!/usr/bin/env python3
# 只保留第二个词组是 L / G / Q 的行
# 用法: python keep_LGQ.py raw.ens cleaned.ens

import argparse
from pathlib import Path
import sys
import pandas as pd
import re
from tqdm import tqdm
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.nucdiff.parser import ENSDFParserV2

def clean_and_merge_files(input_dir: Path, output_file: Path, start_year: int, end_year: int, verbose: bool):
    """
    Cleans and merges ENSDF data files for a given year range.
    Deduplication is now performed per nuclide (Z, A) based on energy proximity and FLAG priority.
    """
    all_levels_dfs = []
    for year in range(start_year, end_year + 1):
        # The parser expects the base name "ENSDF-{year}", and it will handle the different extensions (.L, .G, etc.)
        filepath_base = input_dir / f"ENSDF-{year}"
        # We just need to check if at least one of the possible files exists, e.g., the .L file for levels.
        if (filepath_base.with_suffix('.L')).exists():
            parser = ENSDFParserV2(verbose=verbose)
            # Pass the base path to the parser
            levels_df, gammas_df = parser.parse_file(str(filepath_base))

            if not levels_df.empty:
                # 1. Expert rule filtering: Only keep 'ADOPTED' with specific flags
                # TEMP: Commenting out to debug empty file issue
                # pre_filter_count = len(levels_df)
                # levels_df = levels_df[
                #     (levels_df['dataset_type'] == 'ADOPTED') &
                #     (levels_df['flag'].isin(['A', 'B', ' ']))
                # ].copy()
                # if verbose:
                #     print(f"  - {filepath_base.name}: levels count {pre_filter_count} -> {len(levels_df)} (after ADOPTED filtering)")

                # Attach year information
                levels_df['year'] = year
                all_levels_dfs.append(levels_df)

    if not all_levels_dfs:
        print("No valid level data found in the specified year range.")
        return

    # Combine all data from all years
    combined_df = pd.concat(all_levels_dfs, ignore_index=True)
    print(f"Combined {len(combined_df)} records from {start_year}-{end_year} before final deduplication.")

    # 2. Advanced deduplication based on (Z, A) and energy proximity
    # Define FLAG priority
    flag_priority = {'A': 0, ' ': 1, 'B': 2}
    combined_df['flag_priority'] = combined_df['flag'].map(flag_priority).fillna(3)

    # Sort by year (latest first) and flag priority (highest first)
    # This ensures that when we drop duplicates, we keep the record from the most recent year with the best flag.
    combined_df.sort_values(by=['year', 'flag_priority'], ascending=[False, True], inplace=True)

    # Group by nuclide and apply custom deduplication
    final_records = []
    # Use a tolerance for energy levels (e.g., 0.5 keV)
    energy_tolerance_keV = 0.5

    for (z, a), group in tqdm(combined_df.groupby(['Z', 'A']), desc="Deduplicating Nuclides", ncols=80):
        if group.empty:
            continue

        # Round energy to the nearest tolerance to group similar levels
        group['energy_rounded'] = (group['energy_keV'] / energy_tolerance_keV).round() * energy_tolerance_keV
        
        # Within each energy-grouped nuclide, keep the first record (which is the best due to prior sorting)
        deduplicated_group = group.drop_duplicates(subset=['energy_rounded'], keep='first')
        
        final_records.append(deduplicated_group)

    if not final_records:
        print("No records remained after deduplication.")
        final_df = pd.DataFrame()
    else:
        final_df = pd.concat(final_records, ignore_index=True)
        final_df.drop(columns=['flag_priority', 'energy_rounded'], inplace=True)
        final_df.sort_values(by=['Z', 'A', 'energy_keV'], inplace=True)

    # Save to a single CSV
    final_df.to_csv(output_file, index=False)
    print(f"Successfully cleaned and merged data into {output_file}")
    print(f"Final dataset contains {len(final_df)} unique level records.")


def main():
    """
    Main function: Handles processing of ENSDF data files based on a specified
    year range, merges them, and exports to a single CSV file.
    This function now acts as a wrapper around the more robust clean_and_merge_files.
    """
    parser = argparse.ArgumentParser(
        description="Process ENSDF data files from a specific year range."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the raw ENSDF data files.")
    parser.add_argument("--start_year", type=int, required=True, help="The starting year of the data to process.")
    parser.add_argument("--end_year", type=int, help="The ending year of the data to process (optional). If not provided, only the start_year is processed.")
    parser.add_argument("--output_dir", type=str, default="clean_data_split", help="Directory to store the cleaned and merged CSV files.")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging output from the parser.")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    start_year = args.start_year
    end_year = args.end_year if args.end_year is not None else start_year
    
    tag = f"{start_year}-{end_year}" if start_year != end_year else str(start_year)
    output_file = output_path / f"levels_{tag}.csv"

    # Call the robust cleaning and merging function
    clean_and_merge_files(
        input_dir=input_path,
        output_file=output_file,
        start_year=start_year,
        end_year=end_year,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
