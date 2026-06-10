#!/usr/bin/env -S python3 -u

import sys

import numpy as np

# ── ASE ──────────────────────────────────────────────────────────────────────
try:
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import read, write
except ImportError:
    print("ASE is needed. Install it with: pip install ase")
    sys.exit(1)


def _read_file(filename: str):

    ase_cell = read(filename, format="extxyz")

    return ase_cell


def subtract_forces(cell_file, forces_file):

    actual_cell = _read_file(cell_file)
    forces_cell = _read_file(forces_file)
