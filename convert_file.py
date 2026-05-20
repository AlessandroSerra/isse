#!/usr/bin/env -S python3 -u

import os
import sys
from argparse import ArgumentParser
from glob import glob

try:
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

        for idx in range(1, Natoms + 1):
            f.write(f"{idx}\t")
            f.write(
                f"{positions[idx][0]:.16f}\t{positions[idx][1]:.16f}\t{positions[idx][2]:.16f}\t"
            )
            f.write(
                f"{forces[idx][0]:.16f}\t{forces[idx][1]:.16f}\t{forces[idx][2]:.16f}\n"
            )


def convert(args, infile_type, outfile_type):
    ase_cell = read(args.input, format=infile_type)

    if args.replicate:
        nx, ny, nz = args.replicate
        supercell = ase_cell.repeat([nx, ny, nz])
        supercell.wrap(eps=1e-12)
        ase_cell = supercell

    if "lammps-data" in outfile_type:
        write_lammps_data(args.output, ase_cell, masses=True)

    elif "alm.lmp" in outfile_type:
        _write_lammps_alamode(ase_cell, args.output)

    elif "extxyz" in outfile_type:
        write(
            args.output,
            ase_cell,
            format=outfile_type,
            columns=["symbols", "positions", "masses"],
        )

    else:
        write(args.output, ase_cell, format=outfile_type)


def derive_output(input_path: str, outfile_type: str) -> str:
    """Derive an output filename from the input path and desired output type."""
    # Map ASE format names to sensible file extensions
    ext_map = {
        "extxyz": "xyz",
        "lammps-data": "lmp",
        "poscar": "POSCAR",
        "cif": "cif",
        "espresso-in": "pwi",
        "espresso-out": "pwo",
        # add more mappings as needed
    }
    ext = ext_map.get(outfile_type, outfile_type)
    base = os.path.splitext(input_path)[0]
    return f"{base}.{ext}"


def main() -> None:
    parser = ArgumentParser(
        description="Script to convert atomic structure file formats."
    )

    parser.add_argument(
        "input",
        type=str,
        nargs="+",  # ← accept one or more inputs
        help="Path(s) to input file(s). Supports glob patterns (e.g. 'file*.xyz')",
    )
    parser.add_argument(
        "output",
        type=str,
        nargs="?",  # ← now optional for multi-file mode
        default=None,
        help="Output file path. Required for single-file conversion; "
        "omit when using glob patterns (output names are derived automatically).",
    )
    parser.add_argument("-it", "--input-type", type=str, default=None)
    parser.add_argument("-ot", "--output-type", type=str, default=None)
    parser.add_argument(
        "-r",
        "--replicate",
        type=int,
        nargs=3,
        metavar=("nx", "ny", "nz"),
        help="Replicate the cell along x, y, z (e.g. -r 3 3 3)",
    )

    args = parser.parse_args()

    # ── Expand globs ──────────────────────────────────────────────────────────
    input_files = []
    for pattern in args.input:
        matches = glob(pattern)
        if not matches:
            print(f"Warning: no files matched pattern '{pattern}', skipping.")
        input_files.extend(sorted(matches))

    if not input_files:
        print("Error: no input files found.")
        sys.exit(1)

    # ── Validate: single-file mode requires an explicit output path ───────────
    if len(input_files) == 1 and args.output:
        pairs = [(input_files[0], args.output)]
    elif len(input_files) > 1 and args.output:
        print(
            "Error: an explicit output path can only be used with a single input file."
        )
        sys.exit(1)
    else:
        # Multi-file (or single without explicit output): derive names
        if not args.output_type:
            print(
                "Error: --output-type/-ot is required when using glob / no explicit output path."
            )
            sys.exit(1)
        pairs = [(f, derive_output(f, args.output_type)) for f in input_files]

    # ── Convert each pair ─────────────────────────────────────────────────────
    for infile, outfile in pairs:
        infile_type = args.input_type or guess_format(infile)
        outfile_type = args.output_type or guess_format(outfile)

        if infile_type not in ioformats:
            print(f"Error: unrecognized input format '{infile_type}' for '{infile}'.")
            print_available_formats()
            sys.exit(1)
        if outfile_type not in ioformats and "alm.lmp" not in outfile_type:
            print(
                f"Error: unrecognized output format '{outfile_type}' for '{outfile}'."
            )
            print_available_formats()
            sys.exit(1)

        print(f"  {infile}  →  {outfile}")
        args.input = infile  # reuse existing convert() without changes
        args.output = outfile
        convert(args, infile_type, outfile_type)


if __name__ == "__main__":
    main()
