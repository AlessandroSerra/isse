from __future__ import annotations

import logging
from typing import cast, overload

import numpy as np
from numpy.typing import NDArray

from ..structures import Atoms

logger = logging.getLogger(__name__)


@overload
def get_scaled_positions(
    atoms_or_positions: Atoms,
) -> NDArray[np.float64]: ...


@overload
def get_scaled_positions(
    atoms_or_positions: NDArray[np.float64],
    cell: NDArray[np.float64],
) -> NDArray[np.float64]: ...


def get_scaled_positions(
    atoms_or_positions: Atoms | NDArray[np.float64],
    cell: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """
    Convert Cartesian atomic positions to scaled coordinates.

    The positions can be provided either through an ``Atoms`` object or
    directly as a NumPy array together with the corresponding cell matrix.

    Parameters
    ----------
    atoms_or_positions : Atoms or numpy.ndarray
        Atomistic configuration or Cartesian positions with shape
        ``(n_atoms, 3)``.
    cell : numpy.ndarray or None, optional
        Cell matrix with shape ``(3, 3)``. Required when Cartesian positions
        are passed directly and omitted when an ``Atoms`` object is provided.

    Returns
    -------
    numpy.ndarray
        Scaled positions with shape ``(n_atoms, 3)``.

    Raises
    ------
    TypeError
        If ``cell`` is provided together with an ``Atoms`` object.
    ValueError
        If positions are absent or the cell is absent or invalid.
    """

    if isinstance(atoms_or_positions, Atoms):
        if cell is not None:
            raise TypeError("cell must not be provided when passing an Atoms object")

        positions = atoms_or_positions.positions
        cell = atoms_or_positions.cell
    else:
        positions = atoms_or_positions

    if positions is None:
        raise ValueError("There are no positions in the Atoms object")

    if cell is None or cell.shape != (3, 3):
        raise ValueError("Cell is absent or has an invalid shape")

    return np.asarray(np.linalg.solve(cell.T, positions.T).T, dtype=np.float64)


def find_primitive_cell(
    atoms: Atoms,
    tolerance: float = 1e-5,
) -> Atoms:
    """
    Determine the primitive cell of an atomistic configuration using spglib.

    Parameters
    ----------
    atoms : Atoms
        Atomistic configuration containing Cartesian positions, cell vectors,
        and chemical symbols.
    tolerance : float, optional
        Symmetry-search tolerance in Cartesian distance units. The default is
        ``1e-5``.

    Returns
    -------
    primitive_cell : numpy.ndarray
        Primitive-cell matrix with shape ``(3, 3)``.
    primitive_positions : numpy.ndarray
        Atomic positions in scaled coordinates relative to the primitive cell,
        with shape ``(n_primitive_atoms, 3)``.
    primitive_numbers : numpy.ndarray
        Integer species identifiers for the atoms in the primitive cell, with
        shape ``(n_primitive_atoms,)``.

    Raises
    ------
    ImportError
        If spglib is not available
    ValueError
        If spglib cannot determine a primitive cell.
    """

    try:
        from spglib import SpgCell, find_primitive
    except ImportError as error:
        raise ImportError("spglib is required to find the primitive cell") from error

    symbols = atoms.symbols
    cell = atoms.cell
    positions = atoms.positions

    if symbols is None:
        raise ValueError("Chemical symbols must be specified")

    if positions is None:
        raise ValueError("Atomic positions must be specified")

    if cell is None or cell.shape != (3, 3):
        raise ValueError("Cell is absent or invalid")

    scaled_positions = get_scaled_positions(positions, cell)

    symbol_to_number = {
        symbol: index for index, symbol in enumerate(dict.fromkeys(symbols), start=1)
    }

    # spglib needs different numbers for different chemical species
    numbers = np.array(
        [symbol_to_number[symbol] for symbol in symbols],
        dtype=np.int32,
    )
    spglib_cell = cast(
        SpgCell,
        (cell, scaled_positions, numbers),
    )

    logger.info(
        f"Searching for primitive cell from {len(numbers)} atoms "
        f"with tolerance {tolerance:.3e}"
    )

    primitive = find_primitive(
        spglib_cell,
        symprec=tolerance,
    )

    if primitive is None:
        raise ValueError("spglib could not determine a primitive cell")

    primitive_cell, primitive_positions, primitive_numbers = primitive

    number_to_symbol = {number: symbol for symbol, number in symbol_to_number.items()}
    primitive_symbols = [number_to_symbol[int(number)] for number in primitive_numbers]

    primitive_cell = np.asarray(primitive_cell, dtype=np.float64)
    primitive_positions = np.asarray(primitive_positions, dtype=np.float64)

    primitive_cartesian_positions = primitive_positions @ primitive_cell

    starting_volume = abs(np.linalg.det(cell))
    primitive_volume = abs(np.linalg.det(primitive_cell))

    logger.debug(
        f"Input cell volume: {primitive_volume:.8f}; "
        f"primitive cell volume: {primitive_volume:.8f}"
    )

    if np.isclose(primitive_volume, starting_volume, rtol=1e-8, atol=0.0):
        raise ValueError(
            "No smaller primitive cell was found."
            "The input structure may already be primitive"
            "or may not retain sufficient symmetry"
        )

    logger.info(
        f"Primitive cell found: {len(numbers)} -> "
        f"{len(primitive_numbers)} atoms, "
        f"volume reduction {starting_volume / primitive_volume:.6f}"
    )
    return Atoms(
        symbols=primitive_symbols,
        cell=primitive_cell,
        positions=primitive_cartesian_positions,
        info=atoms.info.copy(),
    )
