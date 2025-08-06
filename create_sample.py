import argparse
import random
from pathlib import Path
from tqdm import tqdm

def get_core_nucid(line: str) -> str | None:
    """
    Extracts the core nuclide identifier from the first 5 characters of a line.
    Example: ' 82BR ' -> '82BR'
    """
    if len(line) < 5:
        return None
    mass_str = line[:3].strip()
    sym_str = line[3:5].strip()
    if mass_str.isdigit() and sym_str.isalpha():
        # Normalize to a standard format, e.g., "82BR"
        return f"{mass_str}{sym_str.upper()}"
    return None

def create_ensdf_sample(input_file: Path, output_file: Path, num_nuclides: int, seed: int, specific_nuclides: list[str] = None):
    """
    Creates a smaller sample ENSDF file by randomly selecting a number of nuclides
    and extracting all of their associated records.
    """
    print(f"Scanning {input_file.name} to identify all nuclides...")
    
    all_nuclides = set()
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Scanning for nuclides"):
            core_id = get_core_nucid(line)
            if core_id:
                all_nuclides.add(core_id)
    
    if not all_nuclides:
        print("No nuclides found in the file. Exiting.")
        return

    print(f"Found {len(all_nuclides)} unique nuclides.")

    # Select nuclides
    if specific_nuclides:
        print(f"Using specific list of {len(specific_nuclides)} nuclides for the sample.")
        selected_nuclides = set(s.upper() for s in specific_nuclides)
    else:
        print(f"Found {len(all_nuclides)} unique nuclides. Selecting {num_nuclides} randomly.")
        random.seed(seed)
        if num_nuclides >= len(all_nuclides):
            print("Number of requested nuclides is greater than or equal to the total number of nuclides.")
            print("Including all nuclides in the sample.")
            selected_nuclides = all_nuclides
        else:
            selected_nuclides = set(random.sample(sorted(list(all_nuclides)), num_nuclides))
    
    if len(selected_nuclides) < 20:
        print(f"Sample nuclides: {sorted(list(selected_nuclides))}")

    print(f"Extracting records for selected nuclides and writing to {output_file.name}...")
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="Writing sample file"):
            core_id = get_core_nucid(line)
            if core_id and core_id in selected_nuclides:
                f_out.write(line)

    print("\nSample file creation complete.")
    print(f"Output written to: {output_file.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Create a smaller ENSDF sample file by selecting random nuclides.")
    parser.add_argument("--input_file", type=Path, required=True, help="Path to the large, raw ENSDF file (e.g., ENSDF-2010.L).")
    parser.add_argument("--output_dir", type=Path, default=Path("clean data"), help="Directory to save the sample file.")
    parser.add_argument("--num_nuclides", type=int, default=20, help="Number of random nuclides to include in the sample.")
    parser.add_argument("--nuclides", type=str, nargs='+', help="A specific list of nuclide IDs (e.g., 27NA 46TI) to include.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Decide on the output filename
    if args.nuclides:
        # Create a deterministic name for the specific nuclides
        tag = "_".join(sorted(args.nuclides))
        output_file = args.output_dir / f"{args.input_file.stem}_{tag}.sample"
    else:
        output_file = args.output_dir / f"{args.input_file.stem}.sample"

    create_ensdf_sample(args.input_file, output_file, args.num_nuclides, args.seed, args.nuclides)

if __name__ == "__main__":
    main()
