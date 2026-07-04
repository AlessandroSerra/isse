"""Remap an Atom object cartesian positions to the form (cell_index, basis_index)"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from ..structures import Atoms
from .symmetry import find_primitive_cell, get_scaled_positions

logger = logging.getLogger(__name__)


def map_atoms_to_primitive(
    atoms: Atoms,
    primitive_cell: NDArray[np.float64] | None = None,
    basis: NDArray[np.float64] | None = None,
    tolerance: float = 1e-3,
) -> tuple[
    tuple[NDArray[np.int32], NDArray[np.int32]],
    NDArray[np.float64],
]:
    """
    Map atoms to primitive-cell translations and basis sites.

    Each Cartesian atomic position is represented as

    ``position = (cell_index + basis[basis_index]) @ primitive_cell``

    within the specified tolerance. The primitive cell and its basis may be
    supplied explicitly. If both are omitted, they are determined from the
    input structure using spglib.

    Parameters
    ----------
    atoms : Atoms
        Atomistic configuration containing Cartesian positions.
    primitive_cell : numpy.ndarray or None, optional
        Primitive-cell matrix with shape ``(3, 3)``. Cell vectors are stored
        by row. It must be provided together with ``basis``. If omitted, the
        primitive cell is determined automatically using spglib.
    basis : numpy.ndarray or None, optional
        Basis-site positions in scaled coordinates relative to
        ``primitive_cell``, with shape ``(n_basis, 3)``. It must be provided
        together with ``primitive_cell``.
    tolerance : float, optional
        Maximum allowed Cartesian mapping residual, in the same length units
        as the atomic positions and cell vectors. It is also used as the
        spglib symmetry tolerance when the primitive structure is determined
        automatically. The default is ``1e-3``.

    Returns
    -------
    mapping : tuple of numpy.ndarray
        Tuple containing:

        - ``cell_indices`` with shape ``(n_atoms, 3)``, giving the integer
          primitive-cell translation associated with each atom;
        - ``basis_indices`` with shape ``(n_atoms,)``, giving the corresponding
          primitive basis-site index.
    residuals : numpy.ndarray
        Cartesian mapping residual for each atom, with shape ``(n_atoms,)``.

    Raises
    ------
    ValueError
        If atomic positions are absent, only one of ``primitive_cell`` and
        ``basis`` is supplied, the supplied arrays have invalid shapes, the
        atom count is incompatible with the number of basis sites, the mapping
        residual exceeds ``tolerance``, or the basis-site populations are
        inconsistent.
    ImportError
        If the primitive structure must be determined automatically but
        spglib is not installed.
    """
    positions = atoms.positions

    if positions is None:
        raise ValueError("Atoms object contains no positions")

    if (primitive_cell is None) != (basis is None):
        raise ValueError("primitive_cell and basis must be provided together")

    if primitive_cell is None and basis is None:
        logger.info(
            "Primitive cell and basis not provided; determining them with spglib"
        )

        primitive_atoms = find_primitive_cell(
            atoms,
            tolerance=tolerance,
        )

        primitive_cell = primitive_atoms.cell
        basis = get_scaled_positions(primitive_atoms)
    else:
        logger.info("Using user-provided primitive cell and basis")

    assert primitive_cell is not None
    assert basis is not None

    positions = np.asarray(positions, dtype=np.float64)
    primitive_cell = np.asarray(primitive_cell, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)

    if primitive_cell.shape != (3, 3):
        raise ValueError(
            f"primitive_cell must have shape (3, 3), found {primitive_cell.shape}"
        )

    if basis.ndim != 2 or basis.shape[1] != 3:
        raise ValueError(f"basis must have shape (n_basis, 3), found {basis.shape}")

    natoms = len(positions)
    nbasis = len(basis)

    if nbasis == 0:
        raise ValueError("basis must contain at least one site")

    if natoms % nbasis != 0:
        raise ValueError(f"{natoms} atoms are not divisible by {nbasis} basis sites")

    scaled_positions = get_scaled_positions(
        positions,
        primitive_cell,
    )

    cell_indices = np.empty((natoms, 3), dtype=np.int32)
    basis_indices = np.empty(natoms, dtype=np.int32)
    residuals = np.empty(natoms, dtype=np.float64)

    for atom_index, scaled_position in enumerate(scaled_positions):
        deltas = scaled_position - basis
        candidate_cells = np.rint(deltas).astype(np.int32)

        fractional_residuals = deltas - candidate_cells
        cartesian_residuals = fractional_residuals @ primitive_cell
        candidate_residuals = np.linalg.norm(
            cartesian_residuals,
            axis=1,
        )

        basis_index = int(np.argmin(candidate_residuals))

        cell_indices[atom_index] = candidate_cells[basis_index]
        basis_indices[atom_index] = basis_index
        residuals[atom_index] = candidate_residuals[basis_index]

    maximum_residual = float(residuals.max())

    if maximum_residual > tolerance:
        raise ValueError(
            "Unreliable primitive mapping: "
            f"maximum residual = {maximum_residual:.6e} "
            f"> {tolerance:.6e}"
        )

    ncells = natoms // nbasis
    counts = np.bincount(
        basis_indices,
        minlength=nbasis,
    )
    expected = np.full(
        nbasis,
        ncells,
        dtype=np.int32,
    )

    if not np.array_equal(counts, expected):
        raise ValueError(
            f"Unexpected basis-site populations: {counts}; expected {expected}"
        )

    logger.info(
        f"Mapped {natoms} atoms onto {nbasis} basis sites "
        f"across {ncells} primitive cells; "
        f"maximum residual {maximum_residual:.3e}"
    )

    return (cell_indices, basis_indices), residuals
