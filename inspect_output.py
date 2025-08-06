import pandas as pd
from pathlib import Path

def inspect_csv_details(file_path: Path):
    """Loads a CSV file and prints a detailed summary for inspection, focusing on textual data."""
    if not file_path.exists():
        print(f"\n--- File Not Found: {file_path.name} ---\n")
        return

    print(f"\n{'='*20} Inspecting: {file_path.name} {'='*20}")
    try:
        # Set pandas display options to show all columns and more text
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)

        df = pd.read_csv(file_path, low_memory=False)

        print(f"\nShape: {df.shape}")
        print("\nColumns:")
        print(df.columns.tolist())

        print("\nHead:")
        print(df.head())

        # --- Detailed Textual Field Inspection ---
        
        # Define key textual columns for each record type
        text_cols_to_check = {
            'l_records': ['spin_parity', 'flags'],
            'g_records': ['multipolarity'],
            'p_records': ['spin_parity', 'ionization_state'],
            'b_records': ['forbiddenness'],
            'e_records': ['forbiddenness'],
            'a_records': [] # No major text fields to check here
        }

        file_key = file_path.name.split('_')[0]
        if file_key in text_cols_to_check:
            for col in text_cols_to_check[file_key]:
                if col in df.columns and not df[col].dropna().empty:
                    print(f"\n--- Analysis of column: '{col}' ---")
                    # Show most common values
                    print("Top 5 most common values:")
                    print(df[col].value_counts().nlargest(5))
                    
                    # Show some unique examples
                    unique_vals = df[col].dropna().unique()
                    print(f"\nTotal unique values: {len(unique_vals)}")
                    print("Sample of unique values:")
                    print(unique_vals[:10])
        
        print(f"{'='* (54 + len(file_path.name))}\n")

    except Exception as e:
        print(f"Error inspecting file: {e}")

def main():
    data_dir = Path("structured_data")
    tag = "2005-2010"

    files_to_inspect = [
        data_dir / f"l_records_{tag}.csv",
        data_dir / f"g_records_{tag}.csv",
        data_dir / f"p_records_{tag}.csv",
        data_dir / f"b_records_{tag}.csv",
        data_dir / f"e_records_{tag}.csv",
        data_dir / f"a_records_{tag}.csv"
    ]
    
    for file in files_to_inspect:
        inspect_csv_details(file)

if __name__ == "__main__":
    main()
