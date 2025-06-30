# ENSDF Data Prep

This repository provides small utilities for working with
[ENSDF](https://www.nndc.bnl.gov/ensdfarchivals/) data files.
It can clean the raw dataset, split records by type and convert
them to Apache Feather format for analysis or machine learning.

## Requirements

- Python 3.8 or later
- [pandas](https://pandas.pydata.org/)

## Basic Usage

1. **Clean a raw ENSDF file to keep only Q/L/G records**

   ```bash
   python clean.py raw.ens cleaned.ens
   ```

2. **Split the cleaned file into individual record types**

   ```bash
   python divide.py cleaned.ens
   ```

3. **Convert the records to Feather format**

   ```bash
   python -m nucdiff.parse_to_feather <data-directory>
   ```

4. **Quick evaluation of saved checkpoints**

   ```bash
   python -m nucdiff.cli.quick_eval \
       --ckpt checkpoints/2023.pt \
       --year 2023 \
       --cfg configs/default.yaml
   ```

Optional: check for missing values in the generated tables

```bash
python check_missing.py
```

The resulting `*.feather` files can then be loaded with pandas for further analysis.

## License

This project is licensed under the [MIT License](LICENSE).
