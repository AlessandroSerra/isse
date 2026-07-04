from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np

from ..constants import symbol_from_mass
from ..structures import Atoms, Trajectory

DUMP_HEADER_NLINES = 9
PS_TO_FS = 1e3
KCAL_MOL_TO_EV = 0.0433641


def parse_lammps_dump(
    filename: str | Path, symbols: Sequence[str] | None = None, units: str = "metal"
) -> Trajectory:
    """
    Parse a LAMMPS dump file as a lazy trajectory.

    LAMMPS dump quantities are interpreted using the following units:
        - time: fs
        - positions and cell vectors: angstrom
        - velocities: angstrom / fs
        - forces: eV / angstrom
        - masses: atomic mass units

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the LAMMPS dump file.
    symbols : Sequence[str] or None, optional
        Chemical symbols in LAMMPS atom-type order. For example,
        ``["Si", "O"]`` maps type 1 to Si and type 2 to O. Required when
        the dump contains atom types but does not contain atomic masses.

    Returns
    -------
    Trajectory
        Lazy trajectory providing access to individual frames.

    Raises
    ------
    ValueError
        If a frame header is incomplete, if no valid frames are found, or if
        chemical symbols cannot be inferred from masses or atom types.
    """

    filepath = Path(filename)
    offsets: list[int] = []
    first_properties: list[str] | None = None

    with filepath.open("rb") as f:
        while True:
            offset = f.tell()
            first_line = f.readline()

            if not first_line:
                break

            if not first_line.startswith(b"ITEM: TIMESTEP"):
                raise ValueError(f"Invalid LAMMPS frame at byte offset {offset}")

            header = [first_line] + [
                f.readline() for _ in range(DUMP_HEADER_NLINES - 1)
            ]

            if any(not line for line in header):
                raise ValueError(f"Incomplete LAMMPS header at byte offset {offset}")

            natoms = int(header[3])
            offsets.append(offset)

            if first_properties is None:
                first_properties = header[8].decode("utf-8").split()[2:]

            for _ in range(natoms):
                if not f.readline():
                    raise ValueError(f"Incomplete LAMMPS frame at byte offset {offset}")

    if not offsets:
        raise ValueError("No LAMMPS dump frames found")

    assert first_properties is not None

    has_masses = "mass" in first_properties
    has_types = "type" in first_properties

    if not has_masses and not has_types:
        raise ValueError(
            "Cannot determine chemical symbols because neither "
            "'mass' nor 'type' is present"
        )

    if not has_masses and has_types and symbols is None:
        raise ValueError(
            "The dump contains atom types but no masses. "
            "Provide symbols in LAMMPS type order"
        )

    frame_reader = partial(_read_frame_lammps_dump, type_symbols=symbols, units=units)

    return Trajectory(
        path=filepath,
        offsets=offsets,
        reader=frame_reader,
    )


def _parse_lammps_box(
    box_header: str,
    box_lines: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    values = [list(map(float, line.split())) for line in box_lines]

    if all(tilt in box_header.split() for tilt in ("xy", "xz", "yz")):
        xlo_bound, xhi_bound, xy = values[0]
        ylo_bound, yhi_bound, xz = values[1]
        zlo, zhi, yz = values[2]

        xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
        xhi = xhi_bound - max(0.0, xy, xz, xy + xz)

        ylo = ylo_bound - min(0.0, yz)
        yhi = yhi_bound - max(0.0, yz)
    else:
        xlo, xhi = values[0][:2]
        ylo, yhi = values[1][:2]
        zlo, zhi = values[2][:2]

        xy = 0.0
        xz = 0.0
        yz = 0.0

    cell = np.array(
        [
            [xhi - xlo, 0.0, 0.0],
            [xy, yhi - ylo, 0.0],
            [xz, yz, zhi - zlo],
        ],
        dtype=np.float64,
    )

    origin = np.array([xlo, ylo, zlo], dtype=np.float64)

    return cell, origin


def _read_frame_lammps_dump(
    filepath: Path,
    offset: int,
    *,
    type_symbols: Sequence[str] | None = None,
    units: str = "metal",
) -> Atoms:
    """
    Read one frame from a LAMMPS dump file.

    Parameters
    ----------
    filepath : pathlib.Path
        Path to the LAMMPS dump file.
    offset : int
        Byte offset marking the beginning of the frame.
    type_symbols : Sequence[str] or None, optional
        Chemical symbols in LAMMPS atom-type order.
    units: str or None, optional
        LAMMPS units of the dump file, default 'metal'

    Returns
    -------
    Atoms
        Parsed atomistic configuration.
    """

    if units not in ("real", "metal"):
        raise ValueError(
            "Unrecognized unit style, please specify either 'metal' or 'real'"
        )
    else:
        if units == "real":
            time_factor = 1
            energy_factor = KCAL_MOL_TO_EV
        else:
            time_factor = PS_TO_FS
            energy_factor = 1

    with filepath.open("rb") as file:
        file.seek(offset)

        header = [file.readline() for _ in range(DUMP_HEADER_NLINES)]

        if any(not line for line in header):
            raise ValueError(f"Incomplete LAMMPS dump header at offset {offset}")

        if not header[0].startswith(b"ITEM: TIMESTEP"):
            raise ValueError(f"Invalid LAMMPS frame at offset {offset}")

        timestep = int(header[1])
        natoms = int(header[3])

        box_header = header[4].decode("utf-8").strip()
        box_lines = [line.decode("utf-8").strip() for line in header[5:8]]

        properties = header[8].decode("utf-8").split()[2:]

        atom_lines = [file.readline() for _ in range(natoms)]

    if any(not line for line in atom_lines):
        raise ValueError(f"Incomplete LAMMPS dump frame at timestep {timestep}")

    cell, origin = _parse_lammps_box(
        box_header,
        box_lines,
    )

    columns = {name: index for index, name in enumerate(properties)}

    has_positions = all(name in columns for name in ("x", "y", "z"))
    has_scaled_positions = all(name in columns for name in ("xs", "ys", "zs"))
    has_unwrapped_positions = all(name in columns for name in ("xu", "yu", "zu"))
    has_velocities = all(name in columns for name in ("vx", "vy", "vz"))
    has_forces = all(name in columns for name in ("fx", "fy", "fz"))
    has_masses = "mass" in columns
    has_types = "type" in columns
    has_ids = "id" in columns

    if not any(
        (
            has_positions,
            has_scaled_positions,
            has_unwrapped_positions,
        )
    ):
        raise ValueError("No supported position fields found in LAMMPS dump")

    if not has_masses and not has_types:
        raise ValueError(
            "Cannot determine chemical symbols because neither "
            "'mass' nor 'type' is present"
        )

    if not has_masses and type_symbols is None:
        raise ValueError(
            "The dump contains atom types but no masses. "
            "Provide symbols in LAMMPS type order"
        )

    if has_positions:
        position_columns = tuple(columns[name] for name in ("x", "y", "z"))
        positions_raw = np.empty(
            (natoms, 3),
            dtype=np.float64,
        )

    if has_scaled_positions:
        scaled_position_columns = tuple(columns[name] for name in ("xs", "ys", "zs"))
        scaled_positions_raw = np.empty(
            (natoms, 3),
            dtype=np.float64,
        )

    if has_unwrapped_positions:
        unwrapped_position_columns = tuple(columns[name] for name in ("xu", "yu", "zu"))
        unwrapped_positions_raw = np.empty(
            (natoms, 3),
            dtype=np.float64,
        )

    if has_velocities:
        velocity_columns = tuple(columns[name] for name in ("vx", "vy", "vz"))
        velocities = np.empty(
            (natoms, 3),
            dtype=np.float64,
        )

    if has_forces:
        force_columns = tuple(columns[name] for name in ("fx", "fy", "fz"))
        forces = np.empty(
            (natoms, 3),
            dtype=np.float64,
        )

    if has_masses:
        masses = np.empty(
            natoms,
            dtype=np.float64,
        )

    if has_ids:
        ids = np.empty(
            natoms,
            dtype=np.int64,
        )

    if has_types:
        types = np.empty(
            natoms,
            dtype=np.int64,
        )

    atom_symbols: list[str] = []

    for atom_index, line in enumerate(atom_lines):
        values = line.split()

        if len(values) != len(properties):
            raise ValueError(
                "Unexpected number of columns at "
                f"timestep {timestep}, atom row {atom_index}"
            )

        if has_positions:
            positions_raw[atom_index] = [
                float(values[column]) for column in position_columns
            ]

        if has_scaled_positions:
            scaled_positions_raw[atom_index] = [
                float(values[column]) for column in scaled_position_columns
            ]

        if has_unwrapped_positions:
            unwrapped_positions_raw[atom_index] = [
                float(values[column]) for column in unwrapped_position_columns
            ]

        if has_velocities:
            velocities[atom_index] = [
                (float(values[column]) * time_factor) for column in velocity_columns
            ]

        if has_forces:
            forces[atom_index] = [
                (float(values[column]) * energy_factor) for column in force_columns
            ]

        if has_ids:
            ids[atom_index] = int(values[columns["id"]])

        if has_types:
            atom_type = int(values[columns["type"]])
            types[atom_index] = atom_type

        if has_masses:
            mass = float(values[columns["mass"]])
            masses[atom_index] = mass

            symbol = symbol_from_mass(mass)

            if symbol is None:
                raise ValueError(f"Could not infer chemical symbol from mass {mass}")

            atom_symbols.append(symbol)
        else:
            assert type_symbols is not None

            try:
                atom_symbols.append(type_symbols[atom_type - 1])
            except IndexError as error:
                raise ValueError(
                    f"No chemical symbol was provided for LAMMPS atom type {atom_type}"
                ) from error

    if has_positions:
        positions = positions_raw - origin
    elif has_scaled_positions:
        positions = scaled_positions_raw @ cell
    elif has_unwrapped_positions:
        unwrapped_positions = unwrapped_positions_raw - origin
        scaled_positions = np.linalg.solve(
            cell.T,
            unwrapped_positions.T,
        ).T
        positions = (scaled_positions % 1.0) @ cell
    else:
        raise ValueError(
            f"No supported position formato found in {filepath} LAMMPS dump"
        )

    positions = np.asarray(positions, dtype=np.float64)

    if has_unwrapped_positions:
        unwrapped_positions = unwrapped_positions_raw - origin

    arrays: dict[str, np.ndarray] = {}

    if has_ids:
        arrays["id"] = ids

    if has_types:
        arrays["type"] = types

    if has_unwrapped_positions:
        return Atoms(
            symbols=atom_symbols,
            cell=cell,
            positions=positions,
            unwrapped_positions=unwrapped_positions,
            velocities=velocities if has_velocities else None,
            masses=masses if has_masses else None,
            forces=forces if has_forces else None,
            arrays=arrays,
            info={"timestep": str(timestep)},
        )

    return Atoms(
        symbols=atom_symbols,
        cell=cell,
        positions=positions,
        velocities=velocities if has_velocities else None,
        masses=masses if has_masses else None,
        forces=forces if has_forces else None,
        arrays=arrays,
        info={"timestep": str(timestep)},
    )
