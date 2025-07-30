import pandas as pd
import sys

def analyze_parsed_results(filepath):
    """
    Analyzes the parsed levels.csv file to find record counts and their sources.
    """
    try:
        # Explicitly specify dtype for potentially mixed-type columns to avoid warnings
        df = pd.read_csv(filepath, dtype={'spin_parity': str, 'half_life': str, 'source_dataset_id': str})
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}", file=sys.stderr)
        return

    nuclides_to_check = {
        "58NI": ("NI", 58),
        "137CS": ("CS", 137),
        "208PB": ("PB", 208),
        "235U": ("U", 235)
    }
    
    print(f"\n--- Analyzing Parsed Results in {filepath} ---")
    
    for name, (sym, mass) in nuclides_to_check.items():
        try:
            nuclide_df = df[(df['nuclide'] == sym) & (df['A'] == mass)]
            
            if nuclide_df.empty:
                print(f"\nNuclide '{name}': No records found.")
                continue
                
            print(f"\nNuclide '{name}': Found {len(nuclide_df)} total records.")
            
            source_counts = nuclide_df.groupby('source_dataset_id').size()
            
            print("  Breakdown by source dataset:")
            for source, count in source_counts.items():
                print(f"    - {count} records from source dataset '{source}'")
        except Exception as e:
            print(f"An error occurred while processing {name}: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_parsed_results(sys.argv[1])
    else:
        print("Usage: python temp_analyze.py <path_to_levels_csv>", file=sys.stderr)
        analyze_parsed_results("output/levels_2004.csv") # Default path 