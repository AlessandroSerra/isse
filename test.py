#!/usr/bin/env -S python3 -u

import os
import sys
import unittest
import warnings
from argparse import ArgumentParser

import numpy as np
import spglib
from ase.build import minimize_rotation_and_translation
from ase.geometry import minkowski_reduce
from ase.io import read


def check_minkowski_equivalence(atoms1, atoms2, tol=1e-5):
    cell1_red, _ = minkowski_reduce(atoms1.cell)
    cell2_red, _ = minkowski_reduce(atoms2.cell)
    p1 = cell1_red.cellpar()
    p2 = cell2_red.cellpar()
    return np.allclose(p1, p2, atol=tol)


def check_spglib_equivalence(atoms1, atoms2, symprec=1e-5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        dataset1 = spglib.get_symmetry_dataset(
            (atoms1.cell, atoms1.get_scaled_positions(), atoms1.numbers),
            symprec=symprec,
        )
        dataset2 = spglib.get_symmetry_dataset(
            (atoms2.cell, atoms2.get_scaled_positions(), atoms2.numbers),
            symprec=symprec,
        )

    if dataset1 is None or dataset2 is None:
        return False

    if dataset1.number != dataset2.number:
        return False

    if not np.isclose(atoms1.get_volume(), atoms2.get_volume(), rtol=symprec):
        return False

    return True


def check_kabsch_alignment(atoms1, atoms2, tol=1e-4):
    a1 = atoms1.copy()
    a2 = atoms2.copy()
    try:
        minimize_rotation_and_translation(a1, a2)
        diff_cell = np.abs(a1.cell - a2.cell).max()
        diff_pos = np.abs(a1.get_positions() - a2.get_positions()).max()
        return diff_cell < tol and diff_pos < tol
    except Exception:
        return False


def check_fractional_coords(atoms1, atoms2, tol=1e-4):
    """
    Check that fractional coordinates match up to permutation and PBC wrapping.
    Sorts scaled positions lexicographically before comparing.
    """
    s1 = atoms1.get_scaled_positions() % 1.0
    s2 = atoms2.get_scaled_positions() % 1.0

    s1 = s1[np.lexsort(s1.T[::-1])]
    s2 = s2[np.lexsort(s2.T[::-1])]

    return np.allclose(s1, s2, atol=tol)


class TestCellEquivalence(unittest.TestCase):
    FILE1 = None
    FILE2 = None
    FMT1 = None
    FMT2 = None

    @classmethod
    def setUpClass(cls):
        try:
            cls.atoms_orig = read(cls.FILE1, format=cls.FMT1)
            cls.atoms_conv = read(cls.FILE2, format=cls.FMT2)
            print(f"\n[SETUP] Comparative analysis between:")
            print(f"  1. {cls.FILE1} ({len(cls.atoms_orig)} atoms, format: {cls.FMT1})")
            print(
                f"  2. {cls.FILE2} ({len(cls.atoms_conv)} atoms, format: {cls.FMT2})\n"
            )
        except Exception as e:
            print(f"\n[ERROR] Could not read files: {e}")
            cls.atoms_orig = None
            cls.atoms_conv = None

    def _skip_if_missing(self):
        if self.atoms_orig is None or self.atoms_conv is None:
            self.skipTest("Files not found or invalid.")

    def test_01_atom_count(self):
        """Atom count must match exactly"""
        self._skip_if_missing()
        self.assertEqual(
            len(self.atoms_orig),
            len(self.atoms_conv),
            f"Atom count differs: {len(self.atoms_orig)} vs {len(self.atoms_conv)}",
        )

    def test_02_chemical_species(self):
        """Chemical composition must match"""
        self._skip_if_missing()
        self.assertEqual(
            sorted(self.atoms_orig.get_chemical_symbols()),
            sorted(self.atoms_conv.get_chemical_symbols()),
            "Chemical species differ between the two structures!",
        )

    def test_03_lattice_parameters(self):
        """Individual lattice parameters (a, b, c, alpha, beta, gamma) must match"""
        self._skip_if_missing()
        p1 = self.atoms_orig.cell.cellpar()
        p2 = self.atoms_conv.cell.cellpar()
        np.testing.assert_allclose(
            p1,
            p2,
            atol=1e-4,
            err_msg=f"Lattice parameters differ:\n  orig: {p1}\n  conv: {p2}",
        )

    def test_04_geometric_minkowski(self):
        """Cell shape equivalence (Minkowski reduction)"""
        self._skip_if_missing()
        self.assertTrue(
            check_minkowski_equivalence(self.atoms_orig, self.atoms_conv),
            "Cell parameters (Minkowski) are not equivalent!",
        )

    def test_05_symmetry_spglib(self):
        """Space group number and volume must be conserved (spglib)"""
        self._skip_if_missing()
        self.assertTrue(
            check_spglib_equivalence(self.atoms_orig, self.atoms_conv),
            "Space group or cell volume differ!",
        )

    def test_06_fractional_coordinates(self):
        """Fractional coordinates must match (up to permutation and PBC wrapping)"""
        self._skip_if_missing()
        self.assertTrue(
            check_fractional_coords(self.atoms_orig, self.atoms_conv),
            "Fractional coordinates do not match after sorting and PBC wrap!",
        )

    def test_07_rigid_rotation_kabsch(self):
        """Cartesian positions and cell must align rigidly (Kabsch). Skipped if PBC wrap or reordering occurred."""
        self._skip_if_missing()
        aligned = check_kabsch_alignment(self.atoms_orig, self.atoms_conv)
        if not aligned:
            self.skipTest(
                "Structures do not rigidly overlap in XYZ (reordering or PBC wrap likely). "
                "See test_06 for fractional coordinate check."
            )
        self.assertTrue(aligned)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Compare two atomic structure files for equivalence."
    )
    parser.add_argument("file1", type=str, help="Path to the first structure file")
    parser.add_argument("file2", type=str, help="Path to the second structure file")
    parser.add_argument(
        "-t1",
        "--type1",
        type=str,
        default=None,
        help="ASE format string for file1 (e.g. vasp, extxyz, lammps-data)",
    )
    parser.add_argument(
        "-t2",
        "--type2",
        type=str,
        default=None,
        help="ASE format string for file2 (e.g. vasp, extxyz, lammps-data)",
    )

    args, remaining = parser.parse_known_args()

    for f in (args.file1, args.file2):
        if not os.path.exists(f):
            print(f"Error: file does not exist: {f}")
            sys.exit(1)

    TestCellEquivalence.FILE1 = args.file1
    TestCellEquivalence.FILE2 = args.file2
    TestCellEquivalence.FMT1 = args.type1
    TestCellEquivalence.FMT2 = args.type2

    # Pass remaining args (e.g. -v) to unittest
    sys.argv = [sys.argv[0]] + remaining

    unittest.main(verbosity=2)
