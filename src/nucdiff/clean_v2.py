import argparse
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm

# Add project root to sys.path to allow for module imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import the new state machine parser
from src.nucdiff.ensdf_parser_v3 import ENSDFParserV3

def clean_and_export(input_dir: Path, output_dir: Path, start_year: int, end_year: int, verbose: bool):
    """
    Uses ENSDFParserV3 to parse a range of yearly data, performs advanced
    deduplication on level records, and saves the structured data into 
    separate CSV files for each record type.
    """
    parser = ENSDFParserV3(verbose=verbose)
    
    all_records = []
    for year in range(start_year, end_year + 1):
        # The parser's parse_file method is designed to find all relevant files (.L, .G, .Q etc.)
        # for a given year based on the base path.
        file_path_base = input_dir / f"ENSDF-{year}"
        print(f"\n--- Processing year {year} ---")
        
        # Check if any file for this year exists before attempting to parse
        if any(file_path_base.with_suffix(ext).exists() for ext in ['.L', '.G', '.Q', '.B', '.E', '.A', '']):
             records = parser.parse_file(str(file_path_base))
             # Add year information to each record for deduplication and future use
             for rec in records:
                 rec['year'] = year
             all_records.extend(records)
        else:
            print(f"No data files found for {year}, skipping.")

    if not all_records:
        print("No records parsed from the specified year range. Exiting.")
        return

    print("\n--- Consolidating all parsed data ---")
    df = pd.DataFrame(all_records)
    
    # --- Separate DataFrames for each record type ---
    record_types = df['record_type'].unique()
    dataframes: Dict[str, pd.DataFrame] = {}
    for rtype in record_types:
        df_rtype = df[df['record_type'] == rtype].copy()
        # Drop columns that are all NaN for this specific record type to keep CSVs clean
        df_rtype.dropna(axis=1, how='all', inplace=True)
        dataframes[rtype] = df_rtype
        print(f"Created DataFrame for record type '{rtype}' with {len(df_rtype)} records.")

    # --- Keep ALL Level Records across years (no cross-year deduplication) ---
    if 'L' in dataframes:
        print("\n--- Keeping all Level records across years; only lightweight per-year housekeeping ---")
        levels_df = dataframes['L'].copy()

        # Preserve original year information and avoid cross-year overwrites.
        # Optional: If truly identical duplicates exist within the SAME year and SAME rounded energy,
        # we can de-duplicate only within that year to reduce exact duplicates.
        energy_tolerance_keV = 0.5
        levels_df['energy_rounded'] = (pd.to_numeric(levels_df['energy_keV'], errors='coerce') / energy_tolerance_keV).round() * energy_tolerance_keV

        # Ensure sort by nuclide and chronological order to retain temporal structure
        sort_cols = [c for c in ['nuclide_id', 'nuclide', 'Z', 'A'] if c in levels_df.columns]
        levels_df.sort_values(by=sort_cols + ['energy_rounded', 'year'], inplace=True, ascending=True)

        # Lightweight same-year dedup (optional, non-destructive across years)
        if {'year', 'energy_rounded'}.issubset(levels_df.columns):
            before = len(levels_df)
            levels_df = levels_df.drop_duplicates(subset=sort_cols + ['year', 'energy_rounded'], keep='first')
            after = len(levels_df)
            print(f"  - Optional same-year de-dup: {before - after} duplicates removed (kept per-year variations).")

        # Clean temporary columns only; keep year
        levels_df.drop(columns=['energy_rounded'], inplace=True, errors='ignore')
        dataframes['L'] = levels_df.reset_index(drop=True)
        print(f"  - Final kept Level records: {len(dataframes['L'])}")

    # --- Save all dataframes to CSV ---
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{start_year}-{end_year}" if start_year != end_year else str(start_year)
    
    for rtype, df_to_save in dataframes.items():
        if not df_to_save.empty:
            # Generate a clean filename, e.g., "l_records_2005-2010.csv"
            filename = f"{rtype.lower()}_records_{tag}.csv"
            output_path = output_dir / filename
            print(f"Saving {len(df_to_save)} records to {output_path}...")
            df_to_save.to_csv(output_path, index=False)

    print("\n--- New cleaning process complete! ---")

def main():
    parser = argparse.ArgumentParser(description="Clean and structure ENSDF data using the V3 state-machine parser.")
    parser.add_argument("--input_dir", type=Path, default=Path("."), help="Directory containing the raw ENSDF data files (e.g., ENSDF-2010.L).")
    parser.add_argument("--start_year", type=int, required=True, help="The starting year of the data to process.")
    parser.add_argument("--end_year", type=int, help="The ending year (inclusive). Defaults to start_year.")
    parser.add_argument("--output_dir", type=Path, default=Path("structured_data"), help="Directory to store the final structured CSV files.")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging output from the parser.")
    args = parser.parse_args()

    end_year = args.end_year if args.end_year is not None else args.start_year

    clean_and_export(args.input_dir, args.output_dir, args.start_year, end_year, args.verbose)

if __name__ == "__main__":
    main()
