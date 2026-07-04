"""Symmetry operations to find scaled positions, find the primitive cell, ..."""

from __future__ import annotations

import logging
from collections import deque
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
            "No smaller primitive cell was found.\n"
            "The input structure may already be primitive or\n"
            "may not retain sufficient symmetry."
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


def _get_supercell_transofm_matrix(
    supercell: NDArray[np.float64],
    primitive_cell: NDArray[np.float64],
    tolerance: float = 1e-6,
) -> NDArray[np.int32]:
    """
    Determine the integer supercell relating a primitive cell to a supercell.

    The lattice vectors are assumed to be stored by row, according to

    ``supercell = supercell_supercell @ primitive_cell``.

    Parameters
    ----------
    supercell : numpy.ndarray
        Simulation-cell supercell with shape ``(3, 3)``.
    primitive_cell : numpy.ndarray
        Primitive-cell supercell with shape ``(3, 3)``.
    tolerance : float, optional
        Absolute tolerance used to verify that the transformation supercell is
        integer-valued. The default is ``1e-6``.

    Returns
    -------
    numpy.ndarray
        Integer supercell supercell with shape ``(3, 3)`` and dtype
        ``numpy.int32``.

    Raises
    ------
    ValueError
        If either cell has an invalid shape, the primitive cell is singular,
        or the simulation cell is not an integer supercell of the primitive
        cell.
    """
    supercell = np.asarray(supercell, dtype=np.float64)
    primitive_cell = np.asarray(primitive_cell, dtype=np.float64)

    if supercell.shape != (3, 3):
        raise ValueError(f"supercell must have shape (3, 3), found {supercell.shape}")

    if primitive_cell.shape != (3, 3):
        raise ValueError(
            f"primitive_cell must have shape (3, 3), found {primitive_cell.shape}"
        )

    try:
        transformation = np.linalg.solve(
            primitive_cell.T,
            supercell.T,
        ).T
    except np.linalg.LinAlgError as error:
        raise ValueError("primitive_cell is singular") from error

    rounded = np.rint(transformation)

    if not np.allclose(
        transformation,
        rounded,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(
            "The simulation cell is not an integer supercell\nof the primitive cell"
        )

    return np.asarray(rounded, dtype=np.int32)


def _generate_qpoints(
    supercell: NDArray[np.int32],
) -> NDArray[np.float64]:
    """
    Generate q-points commensurate with an integer supercell supercell.

    For lattice vectors stored by row, the commensurate reduced q-points
    satisfy

    ``supercell_supercell.T @ q = integer_vector``.

    The number of distinct q-points is equal to

    ``abs(det(supercell_supercell))``.

    Parameters
    ----------
    supercell_supercell : numpy.ndarray
        Integer supercell supercell with shape ``(3, 3)``.
    centered : bool, optional
        If True, return reduced q-point coordinates in ``[-0.5, 0.5)``.
        Otherwise, return them in ``[0, 1)``. The default is True.
    decimals : int, optional
        Number of decimal places used when identifying equivalent q-points.
        The default is 12.

    Returns
    -------
    numpy.ndarray
        Commensurate reduced q-point coordinates with shape
        ``(n_qpoints, 3)``, where
        ``n_qpoints = abs(det(supercell_supercell))``.

    Raises
    ------
    ValueError
        If the supercell has an invalid shape, is not integer-valued, is
        singular, or the expected number of q-points cannot be generated.
    """

    if supercell.shape != (3, 3):
        raise ValueError(
            f"supercell_supercell must have shape (3, 3), found {supercell.shape}"
        )

    rounded = np.rint(supercell)

    if not np.allclose(supercell, rounded, rtol=0.0, atol=0.0):
        raise ValueError("supercell_supercell must contain integer values")

    supercell = np.asarray(rounded, dtype=np.int32)

    determinant = int(round(np.linalg.det(supercell)))
    nqpoints = abs(determinant)

    if nqpoints == 0:
        raise ValueError("supercell_supercell is singular")

    generators = np.linalg.inv(supercell.T)

    zero = np.zeros(3, dtype=np.float64)
    zero_key: tuple[float, float, float] = (
        float(zero[0]),
        float(zero[1]),
        float(zero[2]),
    )

    qpoints_by_key: dict[
        tuple[float, float, float],
        NDArray[np.float64],
    ] = {zero_key: zero}

    pending = deque([zero])

    while pending:
        qpoint = pending.popleft()

        for generator in generators.T:
            candidate = (qpoint + generator) % 1.0
            candidate = np.round(candidate, decimals=10)
            candidate %= 1.0

            key: tuple[float, float, float] = (
                float(candidate[0]),
                float(candidate[1]),
                float(candidate[2]),
            )
            if key not in qpoints_by_key:
                qpoints_by_key[key] = candidate
                pending.append(candidate)

    if len(qpoints_by_key) != nqpoints:
        raise ValueError(
            "Failed to generate the expected number of q-points: "
            f"found {len(qpoints_by_key)}, expected {nqpoints}"
        )

    qpoints = np.asarray(
        list(qpoints_by_key.values()),
        dtype=np.float64,
    )
    qpoints_centered = (qpoints + 0.5) % 1.0 - 0.5
    qpoints_centered = np.round(qpoints_centered, decimals=10)

    order = np.lexsort(
        (
            qpoints_centered[:, 2],
            qpoints_centered[:, 1],
            qpoints_centered[:, 0],
        )
    )

    return np.ascontiguousarray(
        qpoints_centered[order],
        dtype=np.float64,
    )
