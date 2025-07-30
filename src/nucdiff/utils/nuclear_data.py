# src/nucdiff/utils/nuclear_data.py

"""
Central repository for nuclear physics data used in feature engineering.
This module will contain constants, lookup tables, and functions to access
data like magic numbers, binding energies, deformations, etc.
"""

# Magic Numbers for protons and neutrons
# These are the numbers of nucleons such that they are arranged into complete shells
# within the atomic nucleus, resulting in higher stability.
MAGIC_NUMBERS = {2, 8, 20, 28, 50, 82, 126}

# TODO:
# - Add lookup table for experimental binding energies from AME.
# - Add lookup table for theoretical deformations (beta2) from FRDM or similar models.
# - Add functions to calculate pairing energy based on binding energies. 