from __future__ import annotations

from collections.abc import Iterator
from logging import getLogger
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from .helpers.cell_mapping import map_atoms_to_primitive
from .parsers.parse_alamode import read_alamode_evec
from .structures import Atoms, Trajectory

logger = getLogger(__name__)


def _compute_parseval_errors(
    qdot2: NDArray[np.float64],
    atomic_norms: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute the per-frame relative Parseval (completeness) error.

    The modal decomposition is complete and correctly normalized if the sum
    of the squared modal amplitudes over all q-points and modes equals the
    mass-weighted Cartesian velocity norm, for every frame. This function
    quantifies the relative deviation from that identity and can be used to
    detect an incorrect mapping, normalization, or coefficient calculation.

    Parameters
    ----------
    qdot2 : numpy.ndarray
        Squared moduli of the projected modal velocities, with shape
        ``(nframes, nqpoints, nmodes)``.
    atomic_norms : numpy.ndarray
        Mass-weighted Cartesian velocity norm for each frame, with shape
        ``(nframes,)``.

    Returns
    -------
    numpy.ndarray
        Relative Parseval error for each frame, with shape ``(nframes,)``.
        Frames with zero atomic norm are assigned ``numpy.nan``.
    """
    modal_norms = qdot2.sum(axis=(1, 2))
    errors = np.full_like(atomic_norms, np.nan)

    nonzero = atomic_norms != 0.0
    errors[nonzero] = (
        np.abs(modal_norms[nonzero] - atomic_norms[nonzero]) / atomic_norms[nonzero]
    )
    return errors


def _precompute_coefficients(
    qpoints: NDArray[np.float64],
    eigenvectors: NDArray[np.float64],
    cell_indices: NDArray[np.int32],
    basis_indices: NDArray[np.int32],
    masses: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """
    Precompute the frame-independent coefficients for velocity projection.

    The coefficients combine the complex-conjugated phonon eigenvectors,
    the phase factors associated with the primitive-cell translations,
    the atomic mass weights, and the normalization by the number of
    primitive cells.

    The resulting array can be contracted with atomic velocities to obtain
    mode-projected velocities for one or more trajectory frames.

    Parameters
    ----------
    qpoints : numpy.ndarray
        Reduced q-point coordinates with shape ``(nqpoints, 3)``.
    eigenvectors : numpy.ndarray
        Complex phonon eigenvectors with shape
        ``(nqpoints, nmodes, nbasis, 3)``.
    cell_indices : numpy.ndarray
        Integer primitive-cell translation indices for each atom, with shape
        ``(natoms, 3)``.
    basis_indices : numpy.ndarray
        Primitive-basis index associated with each atom, with shape
        ``(natoms,)``.
    masses : numpy.ndarray
        Atomic masses with shape ``(natoms,)``.

    Returns
    -------
    numpy.ndarray
        Complex projection coefficients with shape
        ``(nqpoints, nmodes, natoms, 3)`` and dtype ``complex128``.

    Raises
    ------
    ValueError
        If the eigenvectors do not contain three Cartesian components or if
        the total number of atoms is not divisible by the number of atoms in
        the primitive basis.
    """

    _, _, nbasis, ndim = eigenvectors.shape
    natoms = len(masses)

    if ndim != 3:
        raise ValueError(f"Expected three Cartesian dimensions, found {ndim}.")
    if natoms % nbasis != 0:
        raise ValueError(f"{natoms} atoms are not divisible by {nbasis} basis atoms.")

    ncells = natoms // nbasis
    phases = np.exp(-2j * np.pi * (qpoints @ cell_indices.T))
    eig_atoms = np.take(eigenvectors, basis_indices, axis=2)

    coefficients = (
        eig_atoms.conj()
        * phases[:, None, :, None]
        * np.sqrt(masses)[None, None, :, None]
        / np.sqrt(ncells)
    )
    return np.ascontiguousarray(coefficients, dtype=np.complex128)


def _iter_velocity_batches(
    trajectory: Trajectory,
    natoms: int,
    batch_size: int,
) -> Iterator[NDArray[np.float64]]:
    """
    Yield batches of atomic velocities from a lazy trajectory.

    Parameters
    ----------
    trajectory : Trajectory
        Lazy trajectory yielding one ``Atoms`` object per frame.
    natoms : int
        Expected number of atoms in each trajectory frame.
    batch_size : int
        Maximum number of frames included in each batch.

    Yields
    ------
    numpy.ndarray
        Cartesian atomic velocities with shape
        ``(nframes_batch, natoms, 3)``. The final batch may contain fewer
        than ``batch_size`` frames.

    Raises
    ------
    ValueError
        If a frame does not contain velocities or has an unexpected velocity
        array shape.
    """
    batch: list[NDArray[np.float64]] = []

    for iframe, atoms in enumerate(trajectory):
        velocities = atoms.velocities

        if velocities is None:
            raise ValueError(f"Trajectory frame {iframe} does not contain velocities.")

        if velocities.shape != (natoms, 3):
            raise ValueError(
                f"Expected velocities with shape ({natoms}, 3) in frame "
                f"{iframe}, found {velocities.shape}."
            )

        batch.append(velocities)

        if len(batch) == batch_size:
            yield np.ascontiguousarray(batch, dtype=np.float64)
            batch.clear()

    if batch:
        yield np.ascontiguousarray(batch, dtype=np.float64)


if NUMBA_AVAILABLE:

    @njit(cache=True, parallel=True)
    def _project_batch_numba(
        velocities: NDArray[np.float64],
        masses: NDArray[np.float64],
        coefficients: NDArray[np.complex128],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Project atomic velocities onto phonon modes using the Numba backend.

        The projection is evaluated independently for each trajectory frame,
        q-point, and phonon mode. Frames are processed in parallel using
        ``numba.prange``.

        For each mode, the function computes the squared modulus of the
        mass-weighted projected velocity. It also computes the mass-weighted
        Cartesian velocity norm of each frame, which can be used to verify the
        completeness and normalization of the modal decomposition.

        Parameters
        ----------
        velocities : numpy.ndarray
            Cartesian atomic velocities with shape
            ``(nframes, natoms, 3)``.
        masses : numpy.ndarray
            Atomic masses with shape ``(natoms,)``.
        coefficients : numpy.ndarray
            Complex frame-independent projection coefficients with shape
            ``(nqpoints, nmodes, natoms, 3)``.

        Returns
        -------
        qdot2 : numpy.ndarray
            Squared moduli of the projected modal velocities, with shape
            ``(nframes, nqpoints, nmodes)`` and dtype ``float64``.
        atomic_norms : numpy.ndarray
            Mass-weighted Cartesian velocity norm for each frame, with shape
            ``(nframes,)`` and dtype ``float64``.
        """

        nframes, natoms, _ = velocities.shape
        nq, nmodes, _, _ = coefficients.shape

        qdot2 = np.empty((nframes, nq, nmodes), dtype=np.float64)
        atomic_norms = np.empty(nframes, dtype=np.float64)

        for iframe in prange(nframes):
            atomic_norm = 0.0

            for iatom in range(natoms):
                vx = velocities[iframe, iatom, 0]
                vy = velocities[iframe, iatom, 1]
                vz = velocities[iframe, iatom, 2]
                atomic_norm += masses[iatom] * (vx * vx + vy * vy + vz * vz)

            atomic_norms[iframe] = atomic_norm

            for iq in range(nq):
                for imode in range(nmodes):
                    qreal = 0.0
                    qimag = 0.0

                    for iatom in range(natoms):
                        vx = velocities[iframe, iatom, 0]
                        vy = velocities[iframe, iatom, 1]
                        vz = velocities[iframe, iatom, 2]

                        cx = coefficients[iq, imode, iatom, 0]
                        cy = coefficients[iq, imode, iatom, 1]
                        cz = coefficients[iq, imode, iatom, 2]

                        qreal += cx.real * vx + cy.real * vy + cz.real * vz
                        qimag += cx.imag * vx + cy.imag * vy + cz.imag * vz

                    qdot2[iframe, iq, imode] = qreal * qreal + qimag * qimag

        return qdot2, atomic_norms


def _project_batch_numpy(
    velocities: NDArray[np.float64],
    masses: NDArray[np.float64],
    coefficients: NDArray[np.complex128],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Project atomic velocities onto phonon modes using the NumPy backend.

    The projection is evaluated independently for each trajectory frame,
    q-point, and phonon mode. The contraction over atoms and Cartesian
    components is fully vectorized using ``numpy.einsum``.

    For each mode, the function computes the squared modulus of the
    mass-weighted projected velocity. It also computes the mass-weighted
    Cartesian velocity norm of each frame, which can be used to verify the
    completeness and normalization of the modal decomposition.

    Parameters
    ----------
    velocities : numpy.ndarray
        Cartesian atomic velocities with shape
        ``(nframes, natoms, 3)``.
    masses : numpy.ndarray
        Atomic masses with shape ``(natoms,)``.
    coefficients : numpy.ndarray
        Complex frame-independent projection coefficients with shape
        ``(nqpoints, nmodes, natoms, 3)``.

    Returns
    -------
    qdot2 : numpy.ndarray
        Squared moduli of the projected modal velocities, with shape
        ``(nframes, nqpoints, nmodes)`` and dtype ``float64``.
    atomic_norms : numpy.ndarray
        Mass-weighted Cartesian velocity norm for each frame, with shape
        ``(nframes,)`` and dtype ``float64``.
    """

    atomic_norms = np.einsum(
        "a,fad->f",
        masses,
        velocities**2,
        optimize=True,
    )

    q = np.einsum(
        "qmad,fad->fqm",
        coefficients,
        velocities,
        optimize=True,
    )

    qdot2 = (q.real**2 + q.imag**2).astype(np.float64)
    atomic_norms = atomic_norms.astype(np.float64)

    return qdot2, atomic_norms


def project_velocities(
    trajectory: Trajectory,
    reference_atoms: Atoms,
    evec_filepath: str | Path,
    batch_size: int = 100,
    parseval_tolerance: float = 1e-6,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """
    Project trajectory velocities onto the phonon eigenmodes.

    The reference structure is used to determine the atomic masses, map the
    atoms onto the primitive cell, and precompute all frame-independent
    projection coefficients. The trajectory is then processed lazily in
    batches of frames.

    Parameters
    ----------
    trajectory : Trajectory
        Lazy trajectory yielding one ``Atoms`` object per frame.
    reference_atoms : Atoms
        Reference structure corresponding to the supercell used for the
        molecular dynamics simulation.
    evec_filepath : str or pathlib.Path
        Path to the ALAMODE eigenvector file.
    batch_size : int, optional
        Maximum number of trajectory frames processed in each batch.
        The default is ``100``.

    Returns
    -------
    qpoints : numpy.ndarray
        Reduced q-point coordinates, with shape ``(nqpoints, 3)``.
    qdot2 : numpy.ndarray
        Squared moduli of the projected modal velocities, with shape
        ``(nframes, nqpoints, nmodes)``.
    atomic_norms : numpy.ndarray
        Mass-weighted Cartesian velocity norm for each frame, with shape
        ``(nframes,)``.
    parseval_errors : numpy.ndarray
        Relative Parseval error for each frame, with shape ``(nframes,)``.

    Raises
    ------
    FileNotFoundError
        If the ALAMODE eigenvector file does not exist.
    ValueError
        If the reference structure does not contain atomic masses, if
        ``batch_size`` is not positive or if the trajectory contains no frames.
    """
    evec_filepath = Path(evec_filepath)

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, found {batch_size}.")

    if not evec_filepath.is_file():
        raise FileNotFoundError(f"ALAMODE eigenvector file not found: {evec_filepath}")

    if reference_atoms.masses is not None:
        masses = reference_atoms.masses
    elif trajectory[0].masses is not None:
        logger.warning(
            "No masses found in reference atoms. Using masses from trajectory."
        )
    else:
        raise ValueError(
            "The Trajectory or the reference Atoms object must contain atomic masses."
        )

    masses = np.asarray(masses, dtype=np.float64)
    (
        alamode_primitive_cell,
        alamode_qpoints,
        eigenvalues,
        eigenvectors,
    ) = read_alamode_evec(evec_filepath)

    mapping_results, mapping_residuals = map_atoms_to_primitive(reference_atoms)
    cell_indices, basis_indices = mapping_results

    logger.info(
        f"Successfully mapped atoms to primitive cell with {np.max(mapping_residuals):.2e} A max residual."
    )

    coefficients = _precompute_coefficients(
        alamode_qpoints,
        eigenvectors,
        cell_indices,
        basis_indices,
        masses,
    )

    backend = _project_batch_numba if NUMBA_AVAILABLE else _project_batch_numpy

    qdot2_batches: list[NDArray[np.float64]] = []
    atomic_norm_batches: list[NDArray[np.float64]] = []
    parseval_error_batches: list[NDArray[np.float64]] = []
    ibatch = 0

    for velocities in _iter_velocity_batches(
        trajectory,
        natoms=len(reference_atoms),
        batch_size=batch_size,
    ):
        qdot2_batch, atomic_norms_batch = backend(
            velocities,
            masses,
            coefficients,
        )
        errors_batch = _compute_parseval_errors(qdot2_batch, atomic_norms_batch)
        ibatch += 1
        logger.debug(f"Batch {ibatch}: max Parseval error = {np.nanmax(errors_batch)}")

        if (
            parseval_tolerance is not None
            and np.nanmax(errors_batch) > parseval_tolerance
        ):
            raise RuntimeError(
                "Parseval error exceeds tolerance: "
                f"max={np.nanmax(errors_batch):.6e} > {parseval_tolerance:.6e}."
            )

        qdot2_batches.append(qdot2_batch)
        atomic_norm_batches.append(atomic_norms_batch)
        parseval_error_batches.append(errors_batch)

    if not qdot2_batches:
        raise ValueError(
            "No trajectory frames were read. "
            "The trajectory is empty or the parser did not yield any frames."
        )

    return (
        alamode_qpoints,
        np.concatenate(qdot2_batches, axis=0),
        np.concatenate(atomic_norm_batches, axis=0),
        np.concatenate(parseval_error_batches, axis=0),
    )
