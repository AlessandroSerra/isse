import numpy as np

from ..structures import Atoms


def map_atoms_to_primitive(
    atoms: Atoms,
    primitive_cell: np.ndarray,
    basis: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map atoms to primitive translations and user-defined basis sites."""
    positions = atoms.get_positions()
    natoms = len(atoms)
    nbasis = len(basis)

    if basis.ndim != 2 or basis.shape[1] != 3:
        raise ValueError(f"basis must have shape (nbasis, 3), found {basis.shape}.")
    if natoms % nbasis != 0:
        raise ValueError(f"{natoms} atoms are not divisible by {nbasis} basis sites.")

    fractional = positions @ np.linalg.inv(primitive_cell)
    cell_indices = np.empty((natoms, 3), dtype=np.int64)
    basis_indices = np.empty(natoms, dtype=np.int64)
    residuals = np.empty(natoms, dtype=np.float64)

    for iatom, frac in enumerate(fractional):
        best_residual = np.inf
        best_basis = -1
        best_cell = np.zeros(3, dtype=np.int64)

        for ibasis, tau in enumerate(basis):
            delta = frac - tau
            cell = np.rint(delta).astype(np.int64)
            residual = np.linalg.norm(delta - cell)

            if residual < best_residual:
                best_residual = residual
                best_basis = ibasis
                best_cell = cell

        cell_indices[iatom] = best_cell
        basis_indices[iatom] = best_basis
        residuals[iatom] = best_residual

    if residuals.max() > tolerance:
        raise ValueError(
            "Unreliable primitive mapping: "
            f"maximum residual = {residuals.max():.6e} > {tolerance:.6e}."
        )

    ncells = natoms // nbasis
    counts = np.bincount(basis_indices, minlength=nbasis)
    expected = np.full(nbasis, ncells, dtype=np.int64)
    if not np.array_equal(counts, expected):
        raise ValueError(
            f"Unexpected basis-site populations: {counts}; expected {expected}."
        )

    return cell_indices, basis_indices, residuals
