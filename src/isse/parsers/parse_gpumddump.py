from __future__ import annotations

import shlex as shx
from logging import getLogger
from pathlib import Path

import numpy as np

from ..structures import Atoms, Trajectory

logger = getLogger(__name__)

DUMP_HEADER_NLINES = 2


def parse_gpumd_dump(filename: str | Path) -> Trajectory:
    """
    Parse a GPUMD dump file as a lazy trajectory.

    GPUMD dump quantities are interpreted using the following units:
        - time: fs
        - positions and cell vectors: angstrom
        - velocities: angstrom / fs
        - forces: eV / angstrom
        - masses: atomic mass units

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the GPUMD dump file.

    Returns
    -------
    Trajectory
        Lazy trajectory providing access to individual frames.

    Raises
    ------
    ValueError
        if the header lines are Incomplete or if no masses are found
        and no chemical symbol is provided
    """

    filepath = Path(filename)
    offsets: list[int] = []
    first_header: bytes | None = None

    with filepath.open("rb") as f:
        while True:
            offset = f.tell()
            natoms_line = f.readline()

            if not natoms_line:
                break

            try:
                natoms = int(natoms_line)
            except ValueError as error:
                raise ValueError(
                    f"Invalid atom count at byte offset {offset}"
                ) from error

            header_line = f.readline()

            if not header_line:
                raise ValueError(f"Incomplete GPUMD header at byte offset {offset}")

            if b"Lattice" not in header_line or b"Properties" not in header_line:
                raise ValueError(
                    "Cannot parse GPUMD dump without Properties or Lattice"
                )

            if first_header is None:
                first_header = header_line

            offsets.append(offset)

            for _ in range(natoms):
                if not f.readline():
                    raise ValueError(f"Incomplete GPUMD frame at byte offset {offset}")

    if not offsets:
        raise ValueError("No valid frames found in GPUMD dump")

    if first_header is not None and b"Time" not in first_header:
        logger.warning(
            "No Time property found, timestep information will not be available",
        )

    logger.info(f"Succesfully loaded {len(offsets)} frames from {filepath}")

    return Trajectory(
        path=filepath,
        offsets=offsets,
        reader=_read_frame_gpumd_dump,
    )


def _read_frame_gpumd_dump(filepath: Path, offset: int) -> Atoms:
    """
    Read one frame from a GPUMD dump file.

    Parameters
    ----------
    filepath : pathlib.Path
        Path to the GPUMD dump file.
    offset : int
        Byte offset marking the beginning of the frame.

    Returns
    -------
    Atoms
        Parsed atomistic configuration.
    """

    with filepath.open("rb") as file:
        file.seek(offset)

        header = [file.readline() for _ in range(DUMP_HEADER_NLINES)]

        if any(not line for line in header):
            raise ValueError(f"Incomplete GPUMD dump header at offset {offset}")

        if b"Lattice" not in header[1] or b"Properties" not in header[1]:
            raise ValueError(f"Invalid GPUMD frame at offset {offset}")

        n_atoms = int(header[0])

        if not n_atoms or n_atoms < 1:
            raise ValueError("No number of atoms or no atoms in GPUMD dump")

        atom_lines = [file.readline() for _ in range(n_atoms)]

    header_fields = shx.split(header[1].decode("utf-8"))

    lattice_field = next(
        field for field in header_fields if field.startswith("Lattice=")
    ).removeprefix("Lattice=")

    properties_field = next(
        field for field in header_fields if field.startswith("Properties=")
    ).removeprefix("Properties=")

    timestep_field = next(
        (field for field in header_fields if field.startswith("Time=")),
        None,
    )

    if timestep_field is not None:
        timestep = float(timestep_field.removeprefix("Time="))
    else:
        timestep = None

    cell = np.fromstring(lattice_field, dtype=np.float64, sep=" ", count=9).reshape(
        3, 3
    )

    # every property block is always "property":"type":"n_cols"
    properties_full = properties_field.split(":")
    properties = properties_full[::3]
    properties_n_cols = list(map(int, properties_full[2::3]))
    columns = {}
    column = 0

    for name, n_cols in zip(properties, properties_n_cols):
        columns[name] = column
        column += n_cols

    has_velocities = "vel" in columns
    has_forces = "force" in columns
    has_unwrapped_positions = "unwrapped_position" in columns
    has_masses = "mass" in columns
    has_groups = "group" in columns

    if has_velocities:
        velocities = np.empty((n_atoms, 3), dtype=np.float64)

    if has_forces:
        forces = np.empty((n_atoms, 3), dtype=np.float64)

    if has_unwrapped_positions:
        unwrapped_positions = np.empty((n_atoms, 3), dtype=np.float64)

    if has_masses:
        masses = np.empty(n_atoms, dtype=np.float64)

    if has_groups:
        groups = np.empty(n_atoms, dtype=np.float64)

    atom_symbols: list[str] = []
    arrays: dict[str, np.ndarray] = {}

    positions = np.empty((n_atoms, 3), dtype=np.float64)

    n_columns = sum(properties_n_cols)

    for atom_index, line in enumerate(atom_lines):
        values = line.split()

        if len(values) != n_columns:
            raise ValueError("Unexpected number of columns in GPUMD dump file")

        species_col = columns["species"]
        atom_symbols.append(values[species_col].decode())

        pos_col = columns["pos"]
        positions[atom_index] = [
            float(value) for value in values[pos_col : pos_col + 3]
        ]

        if has_velocities:
            vel_col = columns["vel"]
            velocities[atom_index] = [
                float(value) for value in values[vel_col : vel_col + 3]
            ]

        if has_forces:
            force_col = columns["force"]
            forces[atom_index] = [
                float(value) for value in values[force_col : force_col + 3]
            ]

        if has_unwrapped_positions:
            unwrapped_pos_col = columns["unwrapped_position"]
            unwrapped_positions[atom_index] = [
                float(value)
                for value in values[unwrapped_pos_col : unwrapped_pos_col + 3]
            ]

        if has_masses:
            masses[atom_index] = float(values[columns["mass"]])

        if has_groups:
            groups[atom_index] = int(values[columns["group"]])
            arrays["groups"] = groups

    return Atoms(
        symbols=atom_symbols,
        cell=cell,
        positions=positions,
        velocities=velocities if has_velocities else None,
        masses=masses if has_masses else None,
        forces=forces if has_forces else None,
        arrays=arrays,
        info={"timestep": str(timestep)} if timestep is not None else {},
    )
