
def count_levels_in_source(filepath, nuclide_id):
    """
    A simple and robust way to count valid level records for a specific nuclide
    in the original ENSDF source file.
    """
    count = 0
    with open(filepath, 'r', errors='ignore') as f:
        for line in f:
            if len(line) < 80:
                continue

            # Check for nuclide ID at the start of the line
            if line.startswith(nuclide_id):
                # Check for 'L' record type at the correct position
                if line[7].strip() == 'L':
                    # Check if the energy field (cols 10-19) contains a digit
                    energy_field = line[9:19]
                    if any(char.isdigit() for char in energy_field):
                        count += 1
    return count

if __name__ == "__main__":
    nuclides_to_check = {
        "16O": " 16O  ",
        "58NI": " 58NI ",
        "137CS": "137CS ",
        "208PB": "208PB ",
        "235U": "235U "
    }

    print("--- Counting Valid Level Records in Source File (ENSDF-2004) ---")
    for name, nucid in nuclides_to_check.items():
        count = count_levels_in_source("ENSDF-2004", nucid)
        print(f"'{name}': {count} records") 