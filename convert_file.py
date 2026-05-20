#!/usr/bin/env -S python3 -u

import os
import sys
from argparse import ArgumentParser

try:
    from ase.io import read, write
    from ase.io.formats import ioformats
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
    cell_vectors[cell_vectors < thresh] = 0.0

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
            f.write(f"{idx}\t")
            f.write(
                f"{positions[idx][0]:.16f}\t{positions[idx][1]:.16f}\t{positions[idx][2]:.16f}\t"
            )
            f.write(
                f"{forces[idx][0]:.16f}\t{forces[idx][1]:.16f}\t{forces[idx][2]:.16f}\n"
            )


def convert(args, infile_type, outfile_type):
    ase_cell = read(args.input, format=infile_type)

    if "alm.lmp" in outfile_type:
        _write_lammps_alamode(ase_cell, args.output)
        return

    if args.replicate:
        nx, ny, nz = args.replicate
        supercell = ase_cell.repeat([nx, ny, nz])
        supercell.wrap(eps=1e-12)
        write(args.output, supercell, format=outfile_type)
        return

    write(args.output, ase_cell, format=outfile_type)


def main() -> None:
    parser = ArgumentParser(
        description="Script to convert atomic structure file formats."
    )

    # Positional arguments (No flags required)
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input file",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to the output file",
    )

    # Optional type overrides
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

    infile_type = args.input_type if args.input_type else guess_format(args.input)
    if infile_type not in ioformats:
        print(f"Error: Unsupported or unrecognized input format '{infile_type}'.")
        print_available_formats()
        sys.exit(1)

    outfile_type = args.output_type if args.output_type else guess_format(args.output)
    if outfile_type not in ioformats and "alm.lmp" not in outfile_type:
        print(f"Error: Unsupported or unrecognized output format '{outfile_type}'.")
        print_available_formats()
        sys.exit(1)

    convert(args, infile_type, outfile_type)


if __name__ == "__main__":
    main()
