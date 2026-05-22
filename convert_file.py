#!/usr/bin/env -S python3 -u

import os
import sys
from argparse import ArgumentParser
from glob import glob

import numpy as np

try:
    from ase import Atom, Atoms
    from ase.build import sort
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import read, write
    from ase.io.formats import ioformats
    from ase.io.lammpsdata import write_lammps_data
except ImportError:
    print(
        "ASE is needed to use this script. Please install it with your preferred package manager"
    )
    sys.exit(1)


def print_available_formats() -> None:
    """Helper function to print a clean table of all valid ASE formats."""
    print(f"\n{'Available Format':<18} | {'Description'}")
    print("-" * 65)
    for name in sorted(ioformats.keys()):
        fmt = ioformats[name]
        print(f"{name:<18} | {fmt.description}")


def guess_format(filename: str) -> str:
    """Attempts to guess the format based on file extension or name."""
    base = os.path.basename(filename).lower()
    if "poscar" in base or "contcar" in base:
        return "poscar"

    ext = os.path.splitext(base)[1].strip(".")
    if ext == "xyz":
        return "extxyz"
    return ext


def _write_lammps_alamode(ase_cell, outfile) -> None:
    """manually write an alamode-compatible lammps dump file"""
    Natoms = len(ase_cell)
    positions = ase_cell.get_positions()
    forces = ase_cell.get_forces()
    cell_vectors = ase_cell.cell.array
    thresh = 1e-6
    cell_vectors[np.abs(cell_vectors) < thresh] = 0.0

    with open(outfile, "w") as f:
        f.write("ITEM: TIMESTEP\n")
        f.write("0\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{Natoms}\n")
        f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
        f.write(
            f"{cell_vectors[0][0]:.16e} {cell_vectors[0][1]:.16e} {cell_vectors[0][2]:.16e}\n"
        )
        f.write(
            f"{cell_vectors[1][0]:.16e} {cell_vectors[1][1]:.16e} {cell_vectors[1][2]:.16e}\n"
        )
        f.write(
            f"{cell_vectors[2][0]:.16e} {cell_vectors[2][1]:.16e} {cell_vectors[2][2]:.16e}\n"
        )
        f.write("ITEM: ATOMS id xu yu zu fx fy fz\n")

        for idx in range(Natoms):
            f.write(f"{idx + 1}\t")
            f.write(
                f"{positions[idx][0]:.16f}\t{positions[idx][1]:.16f}\t{positions[idx][2]:.16f}\t"
            )
            f.write(
                f"{forces[idx][0]:.16f}\t{forces[idx][1]:.16f}\t{forces[idx][2]:.16f}\n"
            )


def _read_lammps_alamode(infile: str):

    with open(infile) as f:
        lines = f.readlines()

    Natoms = int(lines[3])
    cell_lines = lines[5:8]
    cell = np.array([list(map(float, line.split())) for line in cell_lines])

    positions = np.zeros((Natoms, 3), dtype=np.float64)
    forces = np.zeros((Natoms, 3), dtype=np.float64)

    offset = 9
    atom_lines = lines[offset : offset + Natoms]

    for i, line in enumerate(atom_lines):
        cols = np.array(line.split(), dtype=np.float64)
        positions[i] = cols[1:4]
        forces[i] = cols[4:7]

    atoms = Atoms(positions=positions, cell=cell, pbc=True)
    calc = SinglePointCalculator(atoms, forces=forces)
    atoms.calc = calc

    return atoms


def derive_output(input_path: str, outfile_type: str) -> str:
    """Derive an output filename from the input path and desired output type."""
    ext_map = {
        "extxyz": "xyz",
        "lammps-data": "lmp",
        "poscar": "POSCAR",
        "cif": "cif",
        "espresso-in": "pwi",
        "espresso-out": "pwo",
    }
    ext = ext_map.get(outfile_type, outfile_type)
    base = os.path.splitext(input_path)[0]
    return f"{base}.{ext}"


def convert(args, infile_type, outfile_type):

    if infile_type in ("lammpstrj", "alm.lmp"):
        ase_cell = _read_lammps_alamode(args.input)
    else:
        ase_cell = read(args.input, format=infile_type)

    if args.replicate:
        nx, ny, nz = args.replicate
        supercell = ase_cell.repeat([nx, ny, nz])
        supercell.wrap(eps=1e-12)
        ase_cell = supercell

    if outfile_type in ("vasp", "poscar"):  # sort BEFORE writing
        ase_cell = sort(ase_cell)

    if "lammps-data" in outfile_type:
        write_lammps_data(args.output, ase_cell, masses=True)

    elif outfile_type in ("alm.lmp", "lammpstrj"):
        _write_lammps_alamode(ase_cell, args.output)

    elif "extxyz" in outfile_type:
        ase_cell.set_masses(
            ase_cell.get_masses()
        )  # weird but necessary to populate the "masses" column
        write(
            args.output,
            ase_cell,
            format=outfile_type,
            columns=["symbols", "positions", "masses"],
        )

    else:
        write(args.output, ase_cell, format=outfile_type, direct=True)


def main() -> None:
    parser = ArgumentParser(
        description="Script to convert atomic structure file formats."
    )

    parser.add_argument(
        "input",
        type=str,
        nargs="+",
        help="Path(s) to input file(s). Supports glob patterns (e.g. 'file*.xyz')",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path. Required for single-file conversion; "
        "omit when using glob patterns (output names are derived automatically).",
    )
    parser.add_argument(
        "-it",
        "--input-type",
        type=str,
        default=None,
        help="Force format of the input file. If omitted, guessed from extension.",
    )
    parser.add_argument(
        "-ot",
        "--output-type",
        type=str,
        default=None,
        help="Force format of the output file. If omitted, guessed from extension.",
    )
    parser.add_argument(
        "-r",
        "--replicate",
        type=int,
        nargs=3,
        metavar=("nx", "ny", "nz"),
        help="Generate a supercell by replicating the system along x, y, and z axes (e.g., -r 3 3 3)",
    )

    args = parser.parse_args()

    # Expand glob patterns
    input_files = []
    for pattern in args.input:
        matches = glob(pattern)
        if not matches:
            print(f"Warning: no files matched pattern '{pattern}', skipping.")
        input_files.extend(sorted(matches))

    if not input_files:
        print("Error: no input files found.")
        sys.exit(1)

    # Single file with explicit output name
    if len(input_files) == 1 and args.output:
        pairs = [(input_files[0], args.output)]

    # Multiple files with explicit output name: not allowed
    elif len(input_files) > 1 and args.output:
        print("Error: -o/--output can only be used with a single input file.")
        sys.exit(1)

    # No explicit output: derive names automatically
    else:
        if not args.output_type:
            print(
                "Error: --output-type/-ot is required when no explicit -o output path is given."
            )
            sys.exit(1)
        pairs = [(f, derive_output(f, args.output_type)) for f in input_files]

    # Convert each pair
    for infile, outfile in pairs:
        infile_type = args.input_type or guess_format(infile)
        outfile_type = args.output_type or guess_format(outfile)

        if infile_type not in ioformats:
            print(f"Error: unrecognized input format '{infile_type}' for '{infile}'.")
            print_available_formats()
            sys.exit(1)

        if outfile_type not in ioformats and outfile_type not in ("alm.lmp", "lammpstrj"):
            print(
                f"Error: unrecognized output format '{outfile_type}' for '{outfile}'."
            )
            print_available_formats()
            sys.exit(1)

        print(f"  {infile}  →  {outfile}")
        args.input = infile
        args.output = outfile
        convert(args, infile_type, outfile_type)


if __name__ == "__main__":
    main()
