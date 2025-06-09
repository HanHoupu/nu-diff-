# ENSDF Data Prep

data resourse: https://www.nndc.bnl.gov/ensdfarchivals/

Plan to create a tiny data-prep tool that pulls Q, L and G records from ENSDF files.

## Requirements

- Python 3.8 or later
- [pandas](https://pandas.pydata.org/)

## Basic Usage

Clean an ENSDF file to keep only Q/L/G records:

```bash
python clean.py raw.ens cleaned.ens
```

Split a raw file into separate L/G/Q files:

```bash
python divide.py raw.ens
```

Convert the cleaned records to Feather format:

```bash
python -m nucdiff.parse_to_feather <data-directory>
```
