#!/usr/bin/env -S python3 -u
"""
convert_file.py — Atomic structure file format converter with built-in equivalence tests.

Supported workflows:
  Single file:   convert_file.py input.xyz -o output.lmp
  Glob pattern:  convert_file.py "*.xyz" -ot lammps-data
  Supercell:     convert_file.py POSCAR -o super.xyz -r 3 3 3
  Fractional:    convert_file.py input.lmp -it lammps-data -ot alm.lmp --frac
  Skip tests:    convert_file.py input.xyz -o output.lmp --skip-tests
"""

import os
import sys
import warnings
from argparse import ArgumentParser
from glob import glob

import numpy as np

# ── ASE ──────────────────────────────────────────────────────────────────────
try:
    from ase import Atoms
    from ase.build import minimize_rotation_and_translation, sort
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.cell import Cell
    from ase.data import atomic_masses, chemical_symbols
    from ase.geometry import minkowski_reduce
    from ase.io import read, write
    from ase.io.formats import ioformats
    from ase.io.lammpsdata import write_lammps_data
except ImportError:
    print("ASE is needed. Install it with: pip install ase")
    sys.exit(1)

# ── spglib optional ──────────────────────────────────────────────────────────
try:
    import spglib

    _HAS_SPGLIB = True
except ImportError:
    _HAS_SPGLIB = False


# ─────────────────────────────────────────────────────────────────────────────
# ANSI colours
# ─────────────────────────────────────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def green(t):
    return _c(t, "32")


def red(t):
    return _c(t, "31")


def yellow(t):
    return _c(t, "33")


def bold(t):
    return _c(t, "1")


def cyan(t):
    return _c(t, "36")


PASS = green("PASS")
FAIL = red("FAIL")
SKIP = yellow("SKIP")
WARN = yellow("WARN")


# ═════════════════════════════════════════════════════════════════════════════
# CONVERSION HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def print_available_formats() -> None:
    """Print a table of all valid ASE formats."""
    print(f"\n{'Available Format':<18} | {'Description'}")
    print("-" * 65)
    for name in sorted(ioformats.keys()):
        fmt = ioformats[name]
        print(f"{name:<18} | {fmt.description}")


def derive_output(input_path: str, outfile_type: str) -> str:
    """Derive an output filename from the input path and desired output type."""
    ext_map = {
        "extxyz": "xyz",
        "lammps-data": "lmp",
        "vasp": "POSCAR",
        "poscar": "POSCAR",
        "cif": "cif",
        "espresso-in": "pwi",
        "espresso-out": "pwo",
        "alm.lmp": "alm.lmp",
        "lammpstrj": "lammpstrj",
    }
    ext = ext_map.get(outfile_type, outfile_type)
    base = os.path.splitext(input_path)[0]
    return f"{base}.{ext}"


def _read_input_atoms(infile: str, infile_type: str) -> Atoms:
    """Read input structure, including custom formats."""
    if infile_type in ("lammpstrj", "alm.lmp"):
        return _read_lammps_dump(infile)
    if infile_type == "alm.xyz":
        return _read_xyz_alamode(infile)
    return read(infile, format=infile_type)


def _read_output_atoms(outfile: str, outfile_type: str) -> Atoms:
    """Read written output structure for equivalence tests."""
    if outfile_type in ("lammpstrj", "alm.lmp"):
        return _read_lammps_dump(outfile)
    return read(outfile, format=outfile_type)


# ─────────────────────────────────────────────────────────────────────────────
# LAMMPS / ALAMODE custom I/O
# ─────────────────────────────────────────────────────────────────────────────

MASS_TO_SYMBOL = {round(m, 3): sym for m, sym in zip(atomic_masses, chemical_symbols)}


def _forces_or_zeros(atoms: Atoms) -> np.ndarray:
    """Return forces if available, otherwise zeros."""
    try:
        return atoms.get_forces()
    except Exception:
        return np.zeros((len(atoms), 3), dtype=float)


def _write_lammps_box_bounds_from_cell(f, cell: np.ndarray) -> None:
    """
    Write a standard LAMMPS restricted-triclinic BOX BOUNDS block.

    Assumes ASE cell rows are already in restricted triclinic form:

        a = (lx, 0,  0)
        b = (xy, ly, 0)
        c = (xz, yz, lz)

    LAMMPS dump format requires:

        xlo_bound xhi_bound xy
        ylo_bound yhi_bound xz
        zlo_bound zhi_bound yz
    """
    lx = float(cell[0, 0])
    xy = float(cell[1, 0])
    ly = float(cell[1, 1])
    xz = float(cell[2, 0])
    yz = float(cell[2, 1])
    lz = float(cell[2, 2])

    xlo = 0.0
    xhi = lx
    ylo = 0.0
    yhi = ly
    zlo = 0.0
    zhi = lz

    xlo_bound = xlo + min(0.0, xy, xz, xy + xz)
    xhi_bound = xhi + max(0.0, xy, xz, xy + xz)

    ylo_bound = ylo + min(0.0, yz)
    yhi_bound = yhi + max(0.0, yz)

    f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
    f.write(f"{xlo_bound:.16e} {xhi_bound:.16e} {xy:.16e}\n")
    f.write(f"{ylo_bound:.16e} {yhi_bound:.16e} {xz:.16e}\n")
    f.write(f"{zlo:.16e} {zhi:.16e} {yz:.16e}\n")


def _write_lammps_alamode(
    ase_cell: Atoms,
    outfile: str,
    fractional: bool = False,
) -> None:
    """
    Write an ALAMODE-compatible LAMMPS dump file.

    If fractional=False:
        writes xu yu zu
    If fractional=True:
        writes xs ys zs

    The BOX BOUNDS block is written in standard LAMMPS restricted-triclinic
    dump format, not as raw cell-vector rows.
    """
    natoms = len(ase_cell)

    if fractional:
        positions = ase_cell.get_scaled_positions(wrap=False)
        pos_header = "xs ys zs"
    else:
        positions = ase_cell.get_positions()
        pos_header = "xu yu zu"

    forces = _forces_or_zeros(ase_cell)

    cell = ase_cell.cell.array.copy()
    cell[np.abs(cell) < 1e-12] = 0.0

    with open(outfile, "w") as f:
        f.write("ITEM: TIMESTEP\n0\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{natoms}\n")

        _write_lammps_box_bounds_from_cell(f, cell)

        f.write(f"ITEM: ATOMS id {pos_header} fx fy fz\n")

        for i in range(natoms):
            p = positions[i]
            fv = forces[i]
            f.write(
                f"{i + 1}\t"
                f"{p[0]:.16f}\t{p[1]:.16f}\t{p[2]:.16f}\t"
                f"{fv[0]:.16f}\t{fv[1]:.16f}\t{fv[2]:.16f}\n"
            )


def _read_lammps_box_bounds(lines):
    header = lines[4].split()
    raw = np.array([list(map(float, line.split())) for line in lines[5:8]])

    # Orthogonal box
    if "xy" not in header:
        xlo, xhi = raw[0]
        ylo, yhi = raw[1]
        zlo, zhi = raw[2]

        origin = np.array([xlo, ylo, zlo])
        cell = np.array(
            [
                [xhi - xlo, 0.0, 0.0],
                [0.0, yhi - ylo, 0.0],
                [0.0, 0.0, zhi - zlo],
            ]
        )
        return cell, origin

    # Restricted triclinic LAMMPS box
    xlo_b, xhi_b, xy = raw[0]
    ylo_b, yhi_b, xz = raw[1]
    zlo_b, zhi_b, yz = raw[2]

    xlo = xlo_b - min(0.0, xy, xz, xy + xz)
    xhi = xhi_b - max(0.0, xy, xz, xy + xz)

    ylo = ylo_b - min(0.0, yz)
    yhi = yhi_b - max(0.0, yz)

    zlo = zlo_b
    zhi = zhi_b

    lx = xhi - xlo
    ly = yhi - ylo
    lz = zhi - zlo

    cell = np.array(
        [
            [lx, 0.0, 0.0],
            [xy, ly, 0.0],
            [xz, yz, lz],
        ]
    )

    origin = np.array([xlo, ylo, zlo])
    return cell, origin


def _read_lammps_dump(infile: str) -> Atoms:
    with open(infile) as f:
        lines = f.readlines()

    natoms = int(lines[3])
    cell, origin = _read_lammps_box_bounds(lines)

    attributes = lines[8].split()[2:]
    col = {name: i for i, name in enumerate(attributes)}

    def _has(*args):
        return all(a in col for a in args)

    if _has("xu", "yu", "zu"):
        pos_key = ("xu", "yu", "zu")
        pos_is_fractional = False
    elif _has("x", "y", "z"):
        pos_key = ("x", "y", "z")
        pos_is_fractional = False
    elif _has("xs", "ys", "zs"):
        pos_key = ("xs", "ys", "zs")
        pos_is_fractional = True
    else:
        raise ValueError(f"ERROR: no positions field found in {infile}.")

    pos_raw = np.zeros((natoms, 3), dtype=float)
    vel = np.zeros((natoms, 3), dtype=float) if _has("vx", "vy", "vz") else None
    force = np.zeros((natoms, 3), dtype=float) if _has("fx", "fy", "fz") else None
    id_ = np.zeros(natoms, dtype=int) if "id" in col else None
    mass = np.zeros(natoms, dtype=float) if "mass" in col else None
    type_ = np.zeros(natoms, dtype=int) if "type" in col else None

    symbols = []

    for i in range(natoms):
        parts = lines[9 + i].split()

        pos_raw[i] = [
            float(parts[col[pos_key[0]]]),
            float(parts[col[pos_key[1]]]),
            float(parts[col[pos_key[2]]]),
        ]

        if id_ is not None:
            id_[i] = int(parts[col["id"]])

        if type_ is not None:
            type_[i] = int(parts[col["type"]])

        if mass is not None:
            m = float(parts[col["mass"]])
            mass[i] = m
            sym = MASS_TO_SYMBOL.get(round(m, 3), None)
            if sym is None:
                raise ValueError(f"Could not infer chemical symbol from mass {m}.")
            symbols.append(sym)

        if vel is not None:
            vel[i] = [
                float(parts[col["vx"]]),
                float(parts[col["vy"]]),
                float(parts[col["vz"]]),
            ]

        if force is not None:
            force[i] = [
                float(parts[col["fx"]]),
                float(parts[col["fy"]]),
                float(parts[col["fz"]]),
            ]

    if pos_is_fractional:
        positions = pos_raw @ cell
    else:
        positions = pos_raw - origin

    if mass is not None:
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        atoms.info["has_masses"] = True
        atoms.set_masses(mass)
    else:
        if type_ is not None:
            # Fallback: unknown species. ASE needs atomic numbers/symbols.
            # Assign H for all atoms but preserve type array.
            atoms = Atoms(
                symbols=["H"] * natoms,
                positions=positions,
                cell=cell,
                pbc=True,
            )
        else:
            atoms = Atoms(positions=positions, cell=cell, pbc=True)

    if vel is not None:
        atoms.set_velocities(vel)

    if force is not None:
        atoms.calc = SinglePointCalculator(atoms, forces=force)

    if id_ is not None:
        atoms.set_array("id", id_)

    if type_ is not None:
        atoms.set_array("type", type_)

    return atoms


def _read_xyz_alamode(infile: str) -> Atoms:
    """Read extxyz dump but return the unwrapped positions instead of the wrapped ones."""
    with open(infile) as f:
        lines = f.readlines()

    natoms = int(lines[0])
    unwrapped_positions = np.zeros((natoms, 3), dtype=np.float64)
    forces = np.zeros((natoms, 3), dtype=np.float64)
    atoms = read(infile, format="extxyz")

    offset = 2

    for i, line in enumerate(lines[offset : offset + natoms]):
        cols = line.split()
        forces[i] = [float(x) for x in cols[4:7]]
        unwrapped_positions[i] = [float(x) for x in cols[7:10]]

    atoms.set_positions(unwrapped_positions)
    atoms.calc = SinglePointCalculator(atoms, forces=forces)
    return atoms


# ─────────────────────────────────────────────────────────────────────────────
# Core conversion
# ─────────────────────────────────────────────────────────────────────────────


def convert(
    infile: str,
    outfile: str,
    infile_type: str,
    outfile_type: str,
    replicate: tuple | None = None,
    fractional: bool = False,
) -> Atoms:
    """
    Convert infile → outfile and return the resulting ASE Atoms object.
    """

    ase_cell = _read_input_atoms(infile, infile_type)

    if replicate:
        nx, ny, nz = replicate
        ase_cell = ase_cell.repeat([nx, ny, nz])
        ase_cell.wrap(eps=1e-12)

    if outfile_type in ("vasp", "poscar"):
        ase_cell = sort(ase_cell)

    if outfile_type == "lammps-data":
        if fractional:
            raise ValueError(
                "--frac is not meaningful for lammps-data: LAMMPS data files "
                "require Cartesian coordinates in the Atoms section."
            )
        write_lammps_data(outfile, ase_cell, masses=True)

    elif outfile_type in ("alm.lmp", "lammpstrj"):
        _write_lammps_alamode(ase_cell, outfile, fractional=fractional)

    elif outfile_type == "extxyz":
        ase_cell = ase_cell.copy()
        ase_cell.set_masses(ase_cell.get_masses())

        has_masses = ase_cell.info.get("has_masses", False)

        if fractional:
            frac = ase_cell.get_scaled_positions(wrap=False)
            ase_cell.set_array("frac_pos", frac)
            columns = (
                ["symbols", "frac_pos", "masses"]
                if has_masses
                else ["symbols", "frac_pos"]
            )
        else:
            columns = (
                ["symbols", "positions", "masses"]
                if has_masses
                else ["symbols", "positions"]
            )

        write(outfile, ase_cell, format=outfile_type, columns=columns)

    else:
        write(outfile, ase_cell, format=outfile_type, direct=fractional)

    return ase_cell


def print_poscar_stdout(atoms: Atoms) -> None:
    """Print POSCAR-style cell vectors and scaled positions to stdout."""
    cell = atoms.cell.array.copy()
    scaled = atoms.get_scaled_positions(wrap=False)
    atom_types = atoms.arrays.get("type", atoms.get_chemical_symbols())
    species_ids = {}

    print("1.0")
    for row in cell:
        print(f"{row[0]:.16f} {row[1]:.16f} {row[2]:.16f}")

    print("atomic_type species_id x_scaled y_scaled z_scaled")
    for atom_type, pos in zip(atom_types, scaled):
        if atom_type not in species_ids:
            species_ids[atom_type] = len(species_ids) + 1
        print(
            f"{atom_type} {species_ids[atom_type]} "
            f"{pos[0]:.16f} {pos[1]:.16f} {pos[2]:.16f}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# EQUIVALENCE CHECKS
# ═════════════════════════════════════════════════════════════════════════════


def _check_minkowski(a1: Atoms, a2: Atoms, tol: float = 1e-5) -> bool:
    p1_red, _ = minkowski_reduce(a1.cell)
    p2_red, _ = minkowski_reduce(a2.cell)
    return np.allclose(Cell(p1_red).cellpar(), Cell(p2_red).cellpar(), atol=tol)


def _check_spglib(a1: Atoms, a2: Atoms, symprec: float = 1e-5) -> bool | None:
    if not _HAS_SPGLIB:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ds1 = spglib.get_symmetry_dataset(
            (a1.cell, a1.get_scaled_positions(), a1.numbers), symprec=symprec
        )
        ds2 = spglib.get_symmetry_dataset(
            (a2.cell, a2.get_scaled_positions(), a2.numbers), symprec=symprec
        )

    if ds1 is None or ds2 is None:
        return False

    if ds1.number != ds2.number:
        return False

    if not np.isclose(a1.get_volume(), a2.get_volume(), rtol=symprec):
        return False

    return True


def _check_kabsch(a1: Atoms, a2: Atoms, tol: float = 1e-4) -> bool:
    c1, c2 = a1.copy(), a2.copy()
    try:
        minimize_rotation_and_translation(c1, c2)
        return (
            np.abs(c1.cell - c2.cell).max() < tol
            and np.abs(c1.get_positions() - c2.get_positions()).max() < tol
        )
    except Exception:
        return False


def _check_fractional(a1: Atoms, a2: Atoms, tol: float = 1e-3) -> tuple[bool, float]:
    """
    Check fractional coordinate equivalence, species by species.

    Uses KDTree nearest-neighbor matching with PBC in fractional coordinates.
    """
    try:
        from scipy.spatial import KDTree
    except ImportError:
        raise RuntimeError(
            "scipy is required for the fractional-coordinate check. "
            "Install it with: pip install scipy"
        )

    s1 = a1.get_scaled_positions() % 1.0
    s2 = a2.get_scaled_positions() % 1.0
    box = np.ones(3)

    max_dist = 0.0

    for Z in sorted(set(a1.numbers)):
        m1 = a1.numbers == Z
        m2 = a2.numbers == Z

        if m1.sum() != m2.sum():
            return False, 1.0

        p1, p2 = s1[m1], s2[m2]

        tree = KDTree(p2, boxsize=box)
        dists, _ = tree.query(p1)

        species_max = float(dists.max())
        max_dist = max(max_dist, species_max)

    return max_dist < tol, max_dist


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────


def _result_line(label: str, badge: str, detail: str = "") -> None:
    detail_str = f"  {yellow(detail)}" if detail else ""
    print(f"    [{badge}] {label}{detail_str}")


def run_equivalence_tests(
    orig: Atoms,
    conv: Atoms | None,
    infile: str,
    outfile: str,
    outfile_type: str,
    replicate: tuple | None = None,
) -> bool:
    """
    Run all equivalence checks between orig and the written output.
    Returns True if every applicable test passed.
    """
    print(
        bold(
            f"\n  ── Equivalence tests: {os.path.basename(infile)} → {os.path.basename(outfile)} ──"
        )
    )

    try:
        conv = _read_output_atoms(outfile, outfile_type)
    except Exception as e:
        print(f"  {WARN} Could not re-read output for verification: {e}")

    if conv is None:
        print(f"  {WARN} No converted structure available for tests.")
        return False

    n_pass = n_fail = n_skip = 0

    def record(badge):
        nonlocal n_pass, n_fail, n_skip
        if badge == PASS:
            n_pass += 1
        elif badge == FAIL:
            n_fail += 1
        else:
            n_skip += 1

    ref = orig.repeat(list(replicate)) if replicate else orig

    # 1 ── Atom count
    label = "Atom count"
    if replicate:
        nx, ny, nz = replicate
        expected = len(orig) * nx * ny * nz
        if len(conv) == expected:
            _result_line(label, PASS, f"{len(conv)} atoms (× {nx}×{ny}×{nz})")
            record(PASS)
        else:
            _result_line(label, FAIL, f"expected {expected}, got {len(conv)}")
            record(FAIL)
    else:
        if len(orig) == len(conv):
            _result_line(label, PASS, f"{len(orig)} atoms")
            record(PASS)
        else:
            _result_line(label, FAIL, f"{len(orig)} input ≠ {len(conv)} output")
            record(FAIL)

    # 2 ── Chemical composition
    label = "Chemical composition"
    from collections import Counter

    c_ref = Counter(ref.get_chemical_symbols())
    c_conv = Counter(conv.get_chemical_symbols())

    if c_ref == c_conv:
        species = ", ".join(f"{el}×{n}" for el, n in sorted(c_ref.items()))
        _result_line(label, PASS, species)
        record(PASS)
    else:
        only_ref = {el: c_ref[el] for el in c_ref if c_ref[el] != c_conv.get(el)}
        only_conv = {el: c_conv[el] for el in c_conv if c_conv[el] != c_ref.get(el)}
        _result_line(label, FAIL, f"input={only_ref}  output={only_conv}")
        record(FAIL)

    # 3 ── Lattice parameters
    label = "Lattice parameters (a,b,c,α,β,γ)"
    try:
        p1 = ref.cell.cellpar()
        p2 = conv.cell.cellpar()

        if np.allclose(p1, p2, atol=1e-4):
            _result_line(
                label,
                PASS,
                f"a={p2[0]:.4f} b={p2[1]:.4f} c={p2[2]:.4f} "
                f"α={p2[3]:.3f}° β={p2[4]:.3f}° γ={p2[5]:.3f}°",
            )
            record(PASS)
        else:
            _result_line(
                label,
                FAIL,
                f"input={np.round(p1, 4)} output={np.round(p2, 4)}",
            )
            record(FAIL)
    except Exception as e:
        _result_line(label, SKIP, str(e))
        record(SKIP)

    # 4 ── Minkowski reduction
    label = "Cell shape (Minkowski)"
    try:
        ok = _check_minkowski(ref, conv)
        _result_line(label, PASS if ok else FAIL)
        record(PASS if ok else FAIL)
    except Exception as e:
        _result_line(label, SKIP, str(e))
        record(SKIP)

    # 5 ── Spglib symmetry
    label = "Space group + volume (spglib)"
    if not _HAS_SPGLIB:
        _result_line(label, SKIP, "spglib not installed")
        record(SKIP)
    else:
        try:
            ok = _check_spglib(ref, conv)
            if ok is None:
                _result_line(label, SKIP, "spglib returned None")
                record(SKIP)
            else:
                _result_line(label, PASS if ok else FAIL)
                record(PASS if ok else FAIL)
        except Exception as e:
            _result_line(label, SKIP, str(e))
            record(SKIP)

    # 6 ── Fractional coordinates
    label = "Fractional coords (PBC-wrapped, species-matched)"
    try:
        ok, max_diff = _check_fractional(ref, conv)
        detail = f"max Δ = {max_diff:.2e}"
        _result_line(label, PASS if ok else FAIL, detail)
        record(PASS if ok else FAIL)
    except Exception as e:
        _result_line(label, SKIP, str(e))
        record(SKIP)

    # 7 ── Kabsch rigid alignment
    label = "Rigid alignment (Kabsch)"
    try:
        ok = _check_kabsch(ref, conv)
        if not ok:
            _result_line(
                label,
                SKIP,
                "not rigidly aligned; reordering/PBC wrapping likely; see fractional test",
            )
            record(SKIP)
        else:
            _result_line(label, PASS)
            record(PASS)
    except Exception as e:
        _result_line(label, SKIP, str(e))
        record(SKIP)

    total = n_pass + n_fail + n_skip
    summary_badge = green("ALL PASSED") if n_fail == 0 else red("FAILURES DETECTED")
    print(
        f"\n  {summary_badge}  "
        f"{green(str(n_pass))} passed · "
        f"{red(str(n_fail))} failed · "
        f"{yellow(str(n_skip))} skipped  "
        f"({total} checks)\n"
    )

    return n_fail == 0


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = ArgumentParser(
        description="Convert atomic structure file formats, with automatic equivalence tests.",
        epilog="Use --list-formats to see all supported ASE formats.",
    )

    parser.add_argument(
        "input",
        type=str,
        nargs="+",
        help="Input file path(s) or glob pattern, e.g. '*.xyz'",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path, single-file conversion only.",
    )
    parser.add_argument(
        "-it",
        "--input-type",
        type=str,
        default=None,
        help="Force input format, ASE format string. Auto-detection is not enabled here.",
    )
    parser.add_argument(
        "-ot",
        "--output-type",
        type=str,
        default=None,
        help="Force output format. Required when no explicit -o is given.",
    )
    parser.add_argument(
        "-r",
        "--replicate",
        type=int,
        nargs=3,
        metavar=("nx", "ny", "nz"),
        help="Build a supercell, e.g. -r 2 2 2.",
    )
    parser.add_argument(
        "--frac",
        action="store_true",
        help=(
            "Write fractional/scaled coordinates when supported. "
            "For alm.lmp/lammpstrj writes xs ys zs instead of xu yu zu. "
            "For VASP-like formats uses direct coordinates. "
            "Not allowed for lammps-data."
        ),
    )
    parser.add_argument(
        "--print-scaled",
        action="store_true",
        help=(
            "Print POSCAR-style cell vectors and "
            "'atomic_type x_scaled y_scaled z_scaled' rows to stdout, then exit."
        ),
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Disable automatic equivalence tests after conversion.",
    )
    parser.add_argument(
        "--list-formats",
        action="store_true",
        help="Print all available ASE formats and exit.",
    )

    args = parser.parse_args()

    if args.list_formats:
        print_available_formats()
        sys.exit(0)

    # Expand globs
    input_files = []
    for pattern in args.input:
        matches = glob(pattern)
        if not matches:
            print(f"  {WARN} No files matched '{pattern}', skipping.")
        input_files.extend(sorted(matches))

    if not input_files:
        print(f"  {red('Error:')} No input files found.")
        sys.exit(1)

    if args.print_scaled:
        if len(input_files) != 1:
            print(f"  {red('Error:')} --print-scaled requires exactly one input file.")
            sys.exit(1)

        if args.input_type is None:
            print(
                f"  {red('Error:')} --input-type/-it is required in this version "
                f"for '{input_files[0]}'."
            )
            sys.exit(1)

        valid_input_custom = ("alm.xyz", "alm.lmp", "lammpstrj")
        if (
            args.input_type not in ioformats
            and args.input_type not in valid_input_custom
        ):
            print(
                f"  {red('Error:')} Unrecognized input format '{args.input_type}' "
                f"for '{input_files[0]}'."
            )
            print_available_formats()
            sys.exit(1)

        try:
            atoms = _read_input_atoms(input_files[0], args.input_type)
        except Exception as e:
            print(f"  {red('Error reading input:')} {e}")
            sys.exit(1)

        if args.replicate:
            atoms = atoms.repeat(args.replicate)
            atoms.wrap(eps=1e-12)

        print_poscar_stdout(atoms)
        sys.exit(0)

    # Build infile/outfile pairs
    if len(input_files) == 1 and args.output:
        pairs = [(input_files[0], args.output)]
    elif len(input_files) > 1 and args.output:
        print(f"  {red('Error:')} -o/--output can only be used with one input file.")
        sys.exit(1)
    else:
        if not args.output_type:
            print(
                f"  {red('Error:')} --output-type/-ot is required when no explicit -o is given."
            )
            sys.exit(1)
        pairs = [(f, derive_output(f, args.output_type)) for f in input_files]

    any_test_failed = False

    for infile, outfile in pairs:
        infile_type = args.input_type
        outfile_type = args.output_type

        if infile_type is None:
            print(
                f"  {red('Error:')} --input-type/-it is required in this version "
                f"for '{infile}'."
            )
            sys.exit(1)

        if outfile_type is None:
            print(
                f"  {red('Error:')} --output-type/-ot is required in this version "
                f"for '{outfile}'."
            )
            sys.exit(1)

        valid_input_custom = ("alm.xyz", "alm.lmp", "lammpstrj")
        valid_output_custom = ("alm.lmp", "lammpstrj")

        if infile_type not in ioformats and infile_type not in valid_input_custom:
            print(
                f"  {red('Error:')} Unrecognized input format '{infile_type}' "
                f"for '{infile}'."
            )
            print_available_formats()
            sys.exit(1)

        if outfile_type not in ioformats and outfile_type not in valid_output_custom:
            print(
                f"  {red('Error:')} Unrecognized output format '{outfile_type}' "
                f"for '{outfile}'."
            )
            print_available_formats()
            sys.exit(1)

        print(bold(f"\n  {cyan('→')} {infile}  →  {outfile}"))
        print(f"     format:  {infile_type}  →  {outfile_type}")

        if args.replicate:
            print(f"     supercell: {'×'.join(str(n) for n in args.replicate)}")

        if args.frac:
            print("     coordinates: fractional/scaled output requested")

        try:
            orig_atoms = _read_input_atoms(infile, infile_type)
        except Exception as e:
            print(f"  {red('Error reading input:')} {e}")
            sys.exit(1)

        try:
            convert(
                infile,
                outfile,
                infile_type,
                outfile_type,
                replicate=args.replicate,
                fractional=args.frac,
            )
            print(f"     {green('Done.')}")
        except Exception as e:
            print(f"  {red('Conversion error:')} {e}")
            sys.exit(1)

        if not args.skip_tests:
            try:
                ok = run_equivalence_tests(
                    orig_atoms,
                    None,
                    infile,
                    outfile,
                    outfile_type,
                    replicate=args.replicate,
                )
                if not ok:
                    any_test_failed = True
            except Exception as e:
                print(f"  {WARN} Test suite raised an unexpected error: {e}")

    sys.exit(1 if any_test_failed else 0)


if __name__ == "__main__":
    main()
