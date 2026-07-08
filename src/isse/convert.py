from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .helpers.periodic import wrap_positions
from .io.parse_gpumddump import parse_gpumd_dump
from .io.parse_lammps import parse_lammps
from .io.parse_vasp import parse_poscar
from .io.write_gpumddump import write_gpumd_dump
from .io.write_lammps import write_lammps_data, write_lammps_dump
from .io.write_vasp import write_poscar
from .structures import Atoms, Trajectory

SUPPORTED_FORMATS = {
    "poscar": "single",
    "vasp": "single",
    "lammps-data": "single",
    "lammps-dump": "trajectory",
    "lammpstrj": "trajectory",
    "gpumd-dump": "trajectory",
    "extxyz": "trajectory",
}


# -----------------------------------------------------------------------------
# Public conversion API
# -----------------------------------------------------------------------------


def convert(
    infile: str | Path,
    outfile: str | Path,
    infile_type: str,
    outfile_type: str,
    *,
    input_units: str | None = None,
    output_units: str | None = None,
    symbols: Sequence[str] | None = None,
    replicate: tuple[int, int, int] | None = None,
    fractional: bool = False,
    frame: int | None = None,
) -> Atoms | Trajectory:
    """
    Convert an atomistic file using only ISSE data structures and I/O modules.

    Supported input formats are those currently parsed by ISSE:
    ``poscar``/``vasp``, ``lammps-data``, ``lammps-dump``/``lammpstrj`` and
    ``gpumd-dump``/``extxyz``. Supported output formats are the same family.

    Parameters
    ----------
    infile : str or pathlib.Path
        Input path.
    outfile : str or pathlib.Path
        Output path.
    infile_type : str
        Explicit input format name.
    outfile_type : str
        Explicit output format name.
    input_units : {"metal", "real"} or None, optional
        LAMMPS input unit style. Required for LAMMPS dump input and used as
        fallback for LAMMPS data input.
    output_units : {"metal", "real"} or None, optional
        LAMMPS output unit style. Required when writing LAMMPS output formats.
    symbols : sequence of str or None, optional
        Chemical symbols in LAMMPS atom-type order when required by I/O modules.
    replicate : tuple of int or None, optional
        Optional ``(nx, ny, nz)`` supercell replication applied to every
        written frame.
    fractional : bool, optional
        Write fractional/scaled coordinates when supported. For POSCAR this
        writes Direct coordinates; for LAMMPS dump this writes ``xs ys zs``.
        LAMMPS data files are always written with Cartesian coordinates.
    frame : int or None, optional
        Select one frame from a trajectory input. Required when writing a
        single-configuration output from a trajectory with more than one frame.

    Returns
    -------
    Atoms or Trajectory
        The parsed input object before writing. Trajectory frames are still
        loaded lazily.
    """
    infile_type = _normalize_format(infile_type)
    outfile_type = _normalize_format(outfile_type)

    parsed = read_file(
        infile,
        infile_type,
        units=input_units,
        symbols=symbols,
    )

    atoms_or_trajectory = _select_data(parsed, frame=frame)
    if replicate is not None:
        atoms_or_trajectory = _replicate_data(atoms_or_trajectory, replicate)

    write_file(
        outfile,
        atoms_or_trajectory,
        outfile_type,
        units=output_units,
        fractional=fractional,
    )

    return parsed


# -----------------------------------------------------------------------------
# Readers and writers
# -----------------------------------------------------------------------------


def read_file(
    filename: str | Path,
    file_type: str,
    *,
    units: str | None = None,
    symbols: Sequence[str] | None = None,
) -> Atoms | Trajectory:
    """
    Read a supported file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path of the input file.
    file_type : str
        Input format name.
    units : {"metal", "real"} or None, optional
        LAMMPS unit style. Required for LAMMPS dump input and used as fallback
        for LAMMPS data input.
    symbols : sequence of str or None, optional
        Chemical symbols in LAMMPS atom-type order when required by the input
        format.

    Returns
    -------
    Atoms or Trajectory
        Parsed atomic configuration or lazy trajectory.
    """
    file_type = _normalize_format(file_type)

    if file_type in ("poscar", "vasp"):
        return parse_poscar(filename)

    if file_type == "lammps-data":
        return parse_lammps(filename, format="data", units=units, symbols=symbols)

    if file_type in ("lammps-dump", "lammpstrj"):
        return parse_lammps(filename, format="dump", units=units, symbols=symbols)

    if file_type in ("gpumd-dump", "extxyz"):
        return parse_gpumd_dump(filename)

    raise ValueError(f"Unsupported input format {file_type!r}.")


def write_file(
    filename: str | Path,
    atoms_or_trajectory: Atoms | Trajectory | list[Atoms],
    file_type: str,
    *,
    units: str | None = None,
    fractional: bool = False,
) -> None:
    """
    Write an atomic configuration or trajectory to a supported format.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path of the output file.
    atoms_or_trajectory : Atoms, Trajectory, or list of Atoms
        Data to write. Single-frame formats require exactly one ``Atoms``
        object. Trajectory formats expect a ``Trajectory``; an in-memory list
        of ``Atoms`` is accepted for generated data such as replicated frames.
    file_type : str
        Output format name.
    units : {"metal", "real"} or None, optional
        LAMMPS unit style for LAMMPS outputs. Required when writing LAMMPS
        output formats.
    fractional : bool, optional
        Write fractional/scaled coordinates where supported. The default is
        ``False``.

    Returns
    -------
    None
        The function writes the file and returns ``None``.
    """
    file_type = _normalize_format(file_type)

    if file_type in ("poscar", "vasp"):
        atoms = _single_atoms(atoms_or_trajectory, file_type)
        write_poscar(filename, atoms, direct=fractional)
        return

    if file_type == "lammps-data":
        atoms = _single_atoms(atoms_or_trajectory, file_type)
        if fractional:
            raise ValueError("fractional=True is not supported for lammps-data output.")
        write_lammps_data(filename, atoms, units=_require_lammps_units(units))
        return

    trajectory = _trajectory_data(atoms_or_trajectory)

    if file_type in ("lammps-dump", "lammpstrj"):
        write_lammps_dump(
            filename,
            trajectory,
            units=_require_lammps_units(units),
            fractional=fractional,
        )
        return

    if file_type in ("gpumd-dump", "extxyz"):
        write_gpumd_dump(filename, trajectory)
        return

    raise ValueError(f"Unsupported output format {file_type!r}.")



# -----------------------------------------------------------------------------
# Equivalence checks
# -----------------------------------------------------------------------------


def run_equivalence_tests(
    original: Atoms | Trajectory,
    output_path: str | Path,
    output_type: str,
    *,
    output_units: str | None = None,
    symbols: Sequence[str] | None = None,
    frame: int | None = None,
    replicate: tuple[int, int, int] | None = None,
) -> bool:
    """
    Run lightweight dependency-free checks on a converted file.

    Parameters
    ----------
    original : Atoms or Trajectory
        Original data before conversion.
    output_path : str or pathlib.Path
        Path of the converted file.
    output_type : str
        Format name of the converted file.
    output_units : {"metal", "real"} or None, optional
        LAMMPS unit style used for reading converted LAMMPS output. Required
        when checking LAMMPS output formats.
    symbols : sequence of str or None, optional
        Chemical symbols in LAMMPS atom-type order when required to read the
        converted file.
    frame : int or None, optional
        Frame selected from the original trajectory during conversion.
    replicate : tuple of int or None, optional
        Replication applied during conversion.

    Returns
    -------
    bool
        ``True`` if all mandatory checks pass, otherwise ``False``.
    """
    output_type = _normalize_format(output_type)

    original_frames = list(_iter_atoms(_select_data(original, frame=frame)))
    if replicate is not None:
        original_frames = [_replicate_atoms(atoms, replicate) for atoms in original_frames]

    read_symbols = symbols
    if read_symbols is None and output_type == "lammps-data" and original_frames:
        _, read_symbols = _type_ids_from_symbols(original_frames[0].symbols)

    if output_type in ("lammps-data", "lammps-dump", "lammpstrj"):
        output_units = _require_lammps_units(output_units)

    converted = read_file(output_path, output_type, units=output_units, symbols=read_symbols)
    converted_frames = list(_iter_atoms(converted))

    ok = True
    if len(original_frames) != len(converted_frames):
        print(f"    [FAIL] frame count: {len(original_frames)} != {len(converted_frames)}")
        return False

    print(f"    [PASS] frame count: {len(converted_frames)}")

    for iframe, (ref, conv) in enumerate(zip(original_frames, converted_frames)):
        prefix = f"frame {iframe}: " if len(original_frames) > 1 else ""
        frame_ok = _check_atoms_equivalent(ref, conv, prefix=prefix)
        ok = ok and frame_ok

    return ok


def _check_atoms_equivalent(ref: Atoms, conv: Atoms, *, prefix: str = "") -> bool:
    ok = True

    if len(ref) == len(conv):
        print(f"    [PASS] {prefix}atom count: {len(ref)}")
    else:
        print(f"    [FAIL] {prefix}atom count: {len(ref)} != {len(conv)}")
        ok = False

    if Counter(ref.symbols) == Counter(conv.symbols):
        print(f"    [PASS] {prefix}composition")
    else:
        print(f"    [FAIL] {prefix}composition")
        ok = False

    if np.allclose(ref.cell, conv.cell, atol=1e-6):
        print(f"    [PASS] {prefix}cell")
    else:
        print(f"    [FAIL] {prefix}cell")
        ok = False

    if np.allclose(_scaled_positions(ref) % 1.0, _scaled_positions(conv) % 1.0, atol=1e-7):
        print(f"    [PASS] {prefix}fractional positions")
    else:
        print(f"    [WARN] {prefix}fractional positions differ or atom order changed")

    return ok


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def derive_output(input_path: str | Path, outfile_type: str) -> str:
    """Derive an output path from an input path and an output format."""
    outfile_type = _normalize_format(outfile_type)
    suffixes = {
        "poscar": "POSCAR",
        "vasp": "POSCAR",
        "lammps-data": "lmp",
        "lammps-dump": "lammpstrj",
        "lammpstrj": "lammpstrj",
        "gpumd-dump": "xyz",
        "extxyz": "xyz",
    }
    path = Path(input_path)
    suffix = suffixes[outfile_type]
    if suffix == "POSCAR":
        return str(path.with_name(path.stem + ".POSCAR"))
    return str(path.with_suffix("." + suffix))


def _require_lammps_units(units: str | None) -> str:
    if units not in ("metal", "real"):
        raise ValueError(
            "LAMMPS units must be specified explicitly as either 'metal' or 'real'."
        )
    return units


def _normalize_format(file_type: str) -> str:
    aliases = {
        "data": "lammps-data",
        "lammps_data": "lammps-data",
        "dump": "lammps-dump",
        "lammps_dump": "lammps-dump",
        "xyz": "gpumd-dump",
        "gpumd": "gpumd-dump",
    }
    normalized = aliases.get(file_type.lower(), file_type.lower())
    if normalized not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format {file_type!r}. Supported formats: "
            + ", ".join(sorted(SUPPORTED_FORMATS))
        )
    return normalized


def _select_data(data: Atoms | Trajectory, *, frame: int | None) -> Atoms | Trajectory:
    if isinstance(data, Atoms):
        if frame is not None and frame != 0:
            raise IndexError("Single-configuration input only has frame 0.")
        return data

    if frame is not None:
        return data[frame]

    return data


def _single_atoms(
    data: Atoms | Trajectory | list[Atoms],
    file_type: str,
) -> Atoms:
    if isinstance(data, Atoms):
        return data

    if isinstance(data, Trajectory):
        if len(data) != 1:
            raise ValueError(
                f"Output format {file_type!r} supports only one frame. Use --frame."
            )
        return data[0]

    if len(data) != 1:
        raise ValueError(f"Output format {file_type!r} supports only one frame. Use --frame.")
    return data[0]


def _trajectory_data(
    data: Atoms | Trajectory | list[Atoms],
) -> Atoms | Trajectory | list[Atoms]:
    return data


def _iter_atoms(data: Atoms | Trajectory | list[Atoms]) -> tuple[Atoms, ...] | list[Atoms] | Trajectory:
    if isinstance(data, Atoms):
        return (data,)
    return data


def _replicate_data(
    data: Atoms | Trajectory,
    replicate: tuple[int, int, int],
) -> Atoms | list[Atoms]:
    if isinstance(data, Atoms):
        return _replicate_atoms(data, replicate)
    return [_replicate_atoms(atoms, replicate) for atoms in data]



def _scaled_positions(atoms: Atoms) -> np.ndarray:
    return np.linalg.solve(atoms.cell.T, atoms.positions.T).T


def _replicate_atoms(atoms: Atoms, replicate: tuple[int, int, int]) -> Atoms:
    if any(n < 1 for n in replicate):
        raise ValueError(f"Replication factors must be positive, found {replicate}.")

    nx, ny, nz = replicate
    shifts = np.array(
        [[ix, iy, iz] for ix in range(nx) for iy in range(ny) for iz in range(nz)],
        dtype=np.float64,
    )
    nrep = len(shifts)
    natoms = len(atoms)
    new_cell = atoms.cell.copy()
    new_cell[0] *= nx
    new_cell[1] *= ny
    new_cell[2] *= nz

    new_positions = np.empty((natoms * nrep, 3), dtype=np.float64)
    new_unwrapped_positions = (
        np.empty((natoms * nrep, 3), dtype=np.float64)
        if atoms.unwrapped_positions is not None
        else None
    )
    for irep, shift in enumerate(shifts):
        start = irep * natoms
        stop = start + natoms
        cartesian_shift = shift @ atoms.cell
        new_positions[start:stop] = atoms.positions + cartesian_shift
        if new_unwrapped_positions is not None:
            assert atoms.unwrapped_positions is not None
            new_unwrapped_positions[start:stop] = atoms.unwrapped_positions + cartesian_shift

    new_arrays = {
        name: np.tile(values, (nrep,) + (1,) * (values.ndim - 1))
        for name, values in atoms.arrays.items()
        if name != "id"
    }
    new_arrays["id"] = np.arange(1, natoms * nrep + 1, dtype=np.int64)

    return Atoms(
        symbols=atoms.symbols * nrep,
        cell=new_cell,
        positions=wrap_positions(new_positions, new_cell),
        unwrapped_positions=new_unwrapped_positions,
        velocities=_tile_optional_atoms_array(atoms.velocities, nrep),
        masses=_tile_optional_atoms_array(atoms.masses, nrep),
        forces=_tile_optional_atoms_array(atoms.forces, nrep),
        arrays=new_arrays,
        info=dict(atoms.info),
    )


def _tile_optional_atoms_array(array: np.ndarray | None, nrep: int) -> np.ndarray | None:
    if array is None:
        return None
    return np.tile(array, (nrep,) + (1,) * (array.ndim - 1))



def _type_ids_from_symbols(symbols: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    type_symbols: list[str] = []
    type_ids = np.empty(len(symbols), dtype=np.int64)
    for index, symbol in enumerate(symbols):
        if symbol not in type_symbols:
            type_symbols.append(symbol)
        type_ids[index] = type_symbols.index(symbol) + 1
    return type_ids, type_symbols


