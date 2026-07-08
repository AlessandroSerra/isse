from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def minimum_image_displacements(
    displacements: NDArray[np.float64],
    cell: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Apply the minimum image convention to Cartesian displacement vectors.

    The cell vectors are stored by row, consistently with the rest of ISSE,
    and Cartesian and fractional coordinates are related by

    ``cartesian = fractional @ cell``.

    Parameters
    ----------
    displacements : numpy.ndarray
        Cartesian displacement vectors with shape ``(..., 3)``.
    cell : numpy.ndarray
        Simulation cell with shape ``(3, 3)``.

    Returns
    -------
    numpy.ndarray
        Minimum-image Cartesian displacement vectors with the same shape as
        ``displacements``.

    Notes
    -----
    This implementation wraps fractional displacements into ``[-0.5, 0.5]``
    using ``numpy.rint``. This is the standard fast minimum-image convention
    for orthogonal and reasonably reduced triclinic cells. For strongly skewed
    cells, the nearest Cartesian image may require checking neighboring cells.
    """
    displacements = np.asarray(displacements, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)

    if displacements.shape[-1] != 3:
        raise ValueError("displacements must have shape (..., 3)")

    if cell.shape != (3, 3):
        raise ValueError(f"cell must have shape (3, 3), found {cell.shape}")

    original_shape = displacements.shape
    flat_displacements = displacements.reshape(-1, 3)

    fractional = np.linalg.solve(cell.T, flat_displacements.T).T
    fractional -= np.rint(fractional)

    minimum_image = fractional @ cell
    return np.asarray(minimum_image, dtype=np.float64).reshape(original_shape)


def minimum_image_distances(
    displacements: NDArray[np.float64],
    cell: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Return minimum-image distances for Cartesian displacement vectors.

    Parameters
    ----------
    displacements : numpy.ndarray
        Cartesian displacement vectors with shape ``(..., 3)``.
    cell : numpy.ndarray
        Simulation cell with shape ``(3, 3)``.

    Returns
    -------
    numpy.ndarray
        Euclidean norms of the minimum-image displacement vectors, with shape
        ``displacements.shape[:-1]``.
    """
    minimum_image = minimum_image_displacements(displacements, cell)
    return np.linalg.norm(minimum_image, axis=-1)


def wrap_positions(
    positions: NDArray[np.float64],
    cell: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Wrap Cartesian positions inside the periodic cell.

    Parameters
    ----------
    positions : numpy.ndarray
        Cartesian positions with shape ``(..., 3)``.
    cell : numpy.ndarray
        Simulation cell with shape ``(3, 3)``.

    Returns
    -------
    numpy.ndarray
        Wrapped Cartesian positions with the same shape as ``positions``.
    """
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)

    if positions.shape[-1] != 3:
        raise ValueError("positions must have shape (..., 3)")

    if cell.shape != (3, 3):
        raise ValueError(f"cell must have shape (3, 3), found {cell.shape}")

    original_shape = positions.shape
    flat_positions = positions.reshape(-1, 3)

    fractional = np.linalg.solve(cell.T, flat_positions.T).T
    fractional %= 1.0

    wrapped = fractional @ cell
    return np.asarray(wrapped, dtype=np.float64).reshape(original_shape)


def unwrap_positions(
    positions: NDArray[np.float64],
    cells: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Unwrap a trajectory of periodically wrapped Cartesian positions.

    The input positions must be ordered in time. The unwrapping is performed
    in fractional coordinates by accumulating minimum-image displacements
    between consecutive frames:

    ``df = f[t] - f[t - 1]``
    ``df -= round(df)``

    Parameters
    ----------
    positions : numpy.ndarray
        Wrapped Cartesian positions with shape ``(n_frames, n_atoms, 3)``.
    cells : numpy.ndarray
        Simulation cell or cells. Accepted shapes are ``(3, 3)`` for a fixed
        cell and ``(n_frames, 3, 3)`` for a time-dependent cell.

    Returns
    -------
    numpy.ndarray
        Unwrapped Cartesian positions with shape ``(n_frames, n_atoms, 3)``.

    Raises
    ------
    ValueError
        If ``positions`` or ``cells`` have incompatible shapes.

    Notes
    -----
    This routine assumes that atoms move by less than half a box length in
    fractional coordinates between consecutive stored frames. If frames are too
    sparse, boundary crossings cannot be reconstructed unambiguously.
    """
    positions = np.asarray(positions, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.float64)

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (n_frames, n_atoms, 3)")

    n_frames = positions.shape[0]

    if cells.shape == (3, 3):
        cells_by_frame = np.broadcast_to(cells, (n_frames, 3, 3))
    elif cells.shape == (n_frames, 3, 3):
        cells_by_frame = cells
    else:
        raise ValueError(
            f"cells must have shape (3, 3) or (n_frames, 3, 3); found {cells.shape}"
        )

    fractional = np.empty_like(positions)

    for iframe in range(n_frames):
        fractional[iframe] = np.linalg.solve(
            cells_by_frame[iframe].T,
            positions[iframe].T,
        ).T

    unwrapped_fractional = np.empty_like(fractional)
    unwrapped_fractional[0] = fractional[0]

    for iframe in range(1, n_frames):
        delta = fractional[iframe] - fractional[iframe - 1]
        delta -= np.rint(delta)
        unwrapped_fractional[iframe] = unwrapped_fractional[iframe - 1] + delta

    unwrapped = np.empty_like(positions)

    for iframe in range(n_frames):
        unwrapped[iframe] = unwrapped_fractional[iframe] @ cells_by_frame[iframe]

    return np.asarray(unwrapped, dtype=np.float64)
