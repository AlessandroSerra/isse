from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .constants import HZ_TO_CM, PS_TO_FS, masses_from_symbols
from .structures import Trajectory

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


TWO_PI = 2.0 * np.pi


def velocity_autocorrelation(
    trajectory: Trajectory,
    *,
    atom_groups: Sequence[int] | NDArray[np.int32] | None = None,
    group: int | None = None,
    max_correlation_len: int | None = None,
    mass_weighted: bool = False,
    remove_com: bool = True,
    time_step: float | None = None,
    batch_size: int = 100,
) -> dict[str, NDArray[np.float64]]:
    """
    Calculate a velocity autocorrelation function using an FFT algorithm.

    Parameters
    ----------
    trajectory
        Lazy :class:`~isse.structures.Trajectory` with at least two frames.
        Each frame must contain velocities. Frames are read one by one while
        building the velocity array required by the FFT.
    atom_groups
        Optional per-atom group labels with shape ``(n_atoms,)``. For example,
        ``[0, 1, 2, 0, 1]`` assigns atoms 0 and 3 to group 0, atoms 1 and 4 to
        group 1, and atom 2 to group 2.
    group
        Group label to select from ``atom_groups``. If ``None``, all atoms are
        used.
    max_correlation_len
        Number of correlation points to return. Defaults to ``n_frames``.
    mass_weighted
        If ``True``, calculate the autocorrelation of ``sqrt(mass) * velocity``.
    remove_com
        If ``True``, remove the instantaneous center-of-mass velocity of all
        atoms before selecting ``group``.
    time_step
        Optional time step between frames, in ISSE internal time units (fs). If
        provided, ``time`` is included in the returned dictionary.
    batch_size
        Number of lazy trajectory frames collected per batch before the FFT.
        The VACF FFT still requires the selected velocity time series to be
        materialized in memory.

    Notes
    -----
    The VACF is always normalized by its value at zero lag and each lag is
    averaged over the corresponding number of available time origins
    ``n_frames - lag``.

    Returns
    -------
    dict
        Contains ``vacf``. Also contains ``time`` when ``time_step`` is provided.
    """
    velocities, masses = _collect_velocities_and_masses(
        trajectory,
        batch_size=batch_size,
    )

    if remove_com:
        velocities = _remove_com_velocity(velocities, masses)
    else:
        velocities = velocities.copy()

    indices = _select_group_indices(
        atom_groups=atom_groups,
        group=group,
        n_atoms=velocities.shape[1],
    )
    if indices is not None:
        velocities = velocities[:, indices, :]
        masses = masses[indices]

    vacf = _vacf_fft_core(
        velocities,
        masses=masses if mass_weighted else None,
        max_correlation_len=max_correlation_len,
    )

    results: dict[str, NDArray[np.float64]] = {"vacf": vacf}
    if time_step is not None:
        if time_step <= 0:
            raise ValueError("time_step must be positive")
        results["time"] = np.arange(vacf.shape[0], dtype=np.float64) * float(time_step)
    return results


def vibrational_density_of_states(
    data: Trajectory | dict[str, NDArray[np.float64]] | Sequence[float] | NDArray[np.floating],
    time_step: float,
    *,
    atom_groups: Sequence[int] | NDArray[np.int32] | None = None,
    group: int | None = None,
    max_correlation_len: int | None = None,
    mass_weighted: bool = False,
    remove_com: bool = True,
    batch_size: int = 100,
    gaussian_filter_width: float | None = None,
) -> dict[str, NDArray[np.float64]]:
    """
    Calculate a vibrational spectrum from a trajectory or from a precomputed VACF.

    Parameters
    ----------
    data
        Either a lazy :class:`~isse.structures.Trajectory`, a VACF array, or a
        dictionary returned by :func:`velocity_autocorrelation` containing
        ``"vacf"``. Any stored time axis is ignored; pass ``time_step`` instead.
    time_step
        Time step between frames or VACF points, in ISSE internal time units
        (fs).
    atom_groups, group, max_correlation_len, mass_weighted, remove_com, batch_size
        Options used only when ``data`` is a trajectory and the VACF must be
        calculated first. The internally calculated VACF is always normalized
        and averaged over the available time origins.
    gaussian_filter_width
        Optional dimensionless Gaussian damping width, following the archived
        Fortran-style implementation.

    Returns
    -------
    dict
        Contains ``frequency`` in cm^-1, ``spectrum``, ``vacf`` and the time
        axis used for the transform. If a Gaussian filter is used, also contains
        ``filtered_vacf`` and ``filter_window``.
    """
    corr, times = _prepare_vdos_input(
        data,
        time_step=time_step,
        atom_groups=atom_groups,
        group=group,
        max_correlation_len=max_correlation_len,
        mass_weighted=mass_weighted,
        remove_com=remove_com,
        batch_size=batch_size,
    )
    if corr.shape[0] < 3:
        raise ValueError("vacf must contain at least three points")

    dt = times[1] - times[0]
    if not np.allclose(np.diff(times), dt, rtol=1.0e-8, atol=1.0e-12):
        raise ValueError("time axis must be uniformly spaced")
    if dt <= 0:
        raise ValueError("time axis must be strictly increasing")

    n_intervals = corr.size - 1
    if n_intervals % 2 != 0:
        n_intervals -= 1
        corr = corr[: n_intervals + 1].copy()
        times = times[: n_intervals + 1]
    else:
        corr = corr.copy()

    if n_intervals < 2:
        raise ValueError("Filon transform requires at least two time intervals")

    dt_ps = float(dt) / PS_TO_FS
    t_max_ps = float(n_intervals) * dt_ps
    delta_omega = 1.0 / t_max_ps

    results: dict[str, NDArray[np.float64]] = {}
    if gaussian_filter_width is not None:
        if gaussian_filter_width < 0:
            raise ValueError("gaussian_filter_width must be non-negative")
        indices = np.arange(n_intervals + 1, dtype=np.float64)
        window = np.exp(
            -0.5 * (0.5 * float(gaussian_filter_width) * indices / n_intervals) ** 2
        )
        corr *= window
        results["filtered_vacf"] = corr.copy()
        results["filter_window"] = window

    spectrum = np.zeros(n_intervals + 1, dtype=np.float64)
    _filon_cosine_transform(
        dt_ps,
        delta_omega,
        n_intervals,
        corr,
        spectrum,
    )

    angular_frequency = np.arange(n_intervals + 1, dtype=np.float64) * delta_omega
    frequency = angular_frequency / TWO_PI * HZ_TO_CM

    results["vacf"] = corr.copy()
    results["time"] = times.copy()
    results["frequency"] = frequency
    results["spectrum"] = spectrum
    return results


def calculate_vdos(
    corr_values: Sequence[float] | NDArray[np.floating],
    time_step: float,
    *,
    gaussian_filter_width: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compatibility wrapper around :func:`vibrational_density_of_states`."""
    results = vibrational_density_of_states(
        corr_values,
        time_step,
        gaussian_filter_width=gaussian_filter_width,
    )
    return results["frequency"], results["spectrum"]


def _prepare_vdos_input(
    data: Trajectory | dict[str, NDArray[np.float64]] | Sequence[float] | NDArray[np.floating],
    *,
    time_step: float,
    atom_groups: Sequence[int] | NDArray[np.int32] | None,
    group: int | None,
    max_correlation_len: int | None,
    mass_weighted: bool,
    remove_com: bool,
    batch_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if isinstance(data, Trajectory):
        vacf_results = velocity_autocorrelation(
            data,
            atom_groups=atom_groups,
            group=group,
            max_correlation_len=max_correlation_len,
            mass_weighted=mass_weighted,
            remove_com=remove_com,
            time_step=time_step,
            batch_size=batch_size,
        )
        corr = np.asarray(vacf_results["vacf"], dtype=np.float64).reshape(-1)
        times = np.asarray(vacf_results["time"], dtype=np.float64).reshape(-1)
        return corr, times

    if isinstance(data, dict):
        if "vacf" not in data:
            raise ValueError("VACF dictionary input must contain a 'vacf' entry")
        corr = np.asarray(data["vacf"], dtype=np.float64).reshape(-1)
        times = _prepare_time_axis(time_step=time_step, n=corr.size)
        return corr, times

    has_vacf_options = (
        any(option is not None for option in (atom_groups, group, max_correlation_len))
        or mass_weighted
        or not remove_com
        or batch_size != 100
    )
    if has_vacf_options:
        raise ValueError("VACF calculation options require trajectory input")

    corr = np.asarray(data, dtype=np.float64).reshape(-1)
    times = _prepare_time_axis(time_step=time_step, n=corr.size)
    return corr, times


def _collect_velocities_and_masses(
    frames: Trajectory,
    *,
    batch_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if len(frames) < 2:
        raise ValueError("trajectory must contain at least two frames")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    first = frames[0]
    reference_symbols = list(first.symbols)
    n_frames = len(frames)
    n_atoms = len(first)

    velocities = np.empty((n_frames, n_atoms, 3), dtype=np.float64)
    masses = (
        np.asarray(first.masses, dtype=np.float64)
        if first.masses is not None
        else np.asarray(masses_from_symbols(first.symbols), dtype=np.float64)
    )
    if masses.shape != (n_atoms,):
        raise ValueError("masses must have shape (n_atoms,)")

    for start in range(0, n_frames, batch_size):
        stop = min(start + batch_size, n_frames)
        for iframe in range(start, stop):
            atoms = first if iframe == 0 else frames[iframe]
            if list(atoms.symbols) != reference_symbols:
                raise ValueError(f"frame {iframe} has a different atom/symbol ordering")
            if atoms.velocities is None:
                raise ValueError(f"frame {iframe} does not contain velocities")
            if atoms.velocities.shape != (n_atoms, 3):
                raise ValueError(f"frame {iframe} velocities have invalid shape")

            frame_masses = (
                np.asarray(atoms.masses, dtype=np.float64)
                if atoms.masses is not None
                else np.asarray(masses_from_symbols(atoms.symbols), dtype=np.float64)
            )
            if frame_masses.shape != (n_atoms,):
                raise ValueError(f"frame {iframe} masses have invalid shape")
            if not np.allclose(frame_masses, masses, rtol=0.0, atol=0.0):
                raise ValueError("frame-dependent masses are not supported")
            velocities[iframe] = atoms.velocities

    return velocities, masses


def _select_group_indices(
    *,
    atom_groups: Sequence[int] | NDArray[np.int32] | None,
    group: int | None,
    n_atoms: int,
) -> NDArray[np.int32] | None:
    if group is None:
        if atom_groups is not None:
            raise ValueError("group must be specified when atom_groups is provided")
        return None

    if atom_groups is None:
        raise ValueError("atom_groups is required when group is specified")

    groups = np.asarray(atom_groups, dtype=np.int32).reshape(-1)
    if groups.shape != (n_atoms,):
        raise ValueError("atom_groups must have shape (n_atoms,)")

    indices = np.nonzero(groups == int(group))[0].astype(np.int32, copy=False)
    if indices.size == 0:
        raise ValueError(f"no atoms found for group {group}")
    return indices


def _remove_com_velocity(
    velocities: NDArray[np.float64], masses: NDArray[np.float64]
) -> NDArray[np.float64]:
    total_mass = np.sum(masses)
    if total_mass <= 0:
        raise ValueError("total mass must be positive")
    com_velocity = np.sum(velocities * masses.reshape(1, -1, 1), axis=1) / total_mass
    return velocities - com_velocity.reshape(-1, 1, 3)


def _vacf_fft_core(
    velocities: NDArray[np.float64],
    *,
    masses: NDArray[np.float64] | None,
    max_correlation_len: int | None,
) -> NDArray[np.float64]:
    vel = np.asarray(velocities, dtype=np.float64)
    if vel.ndim != 3 or vel.shape[2] != 3:
        raise ValueError("velocities must have shape (n_frames, n_atoms, 3)")

    n_frames, n_atoms, ndim = vel.shape
    corr_len = n_frames if max_correlation_len is None else int(max_correlation_len)
    if corr_len < 1:
        raise ValueError("max_correlation_len must be positive")
    if corr_len > n_frames:
        raise ValueError("max_correlation_len cannot exceed the number of frames")

    vel_flat = vel.reshape(n_frames, n_atoms * ndim).copy()
    if masses is not None:
        masses = np.asarray(masses, dtype=np.float64)
        if masses.shape != (n_atoms,):
            raise ValueError("masses must have shape (n_atoms,)")
        vel_flat *= np.sqrt(np.repeat(masses, ndim)).reshape(1, -1)

    n_fft = 1
    while n_fft < 2 * n_frames:
        n_fft <<= 1

    spectrum = np.fft.rfft(vel_flat, n=n_fft, axis=0)
    autocorr = np.fft.irfft(spectrum * np.conjugate(spectrum), n=n_fft, axis=0)[
        :corr_len
    ].real
    vacf = autocorr.sum(axis=1)

    vacf /= np.arange(n_frames, n_frames - corr_len, -1, dtype=np.float64)

    if vacf[0] == 0.0:
        raise ValueError("cannot normalize a VACF with zero value at t=0")
    vacf /= vacf[0]

    return vacf.astype(np.float64, copy=False)


def _prepare_time_axis(
    *,
    time_step: float,
    n: int,
) -> NDArray[np.float64]:
    if time_step <= 0:
        raise ValueError("time_step must be positive")
    return np.arange(n, dtype=np.float64) * float(time_step)


def _filon_cosine_transform_numpy(
    dt: float,
    delta_omega: float,
    n_intervals: int,
    corr: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    if n_intervals % 2 != 0:
        raise ValueError("n_intervals must be even for the Filon transform")
    if corr.shape[0] != n_intervals + 1:
        raise ValueError("corr length must be n_intervals + 1")
    if out.shape[0] != n_intervals + 1:
        raise ValueError("out length must be n_intervals + 1")

    t_max = float(n_intervals) * dt
    nu = np.arange(n_intervals + 1, dtype=np.float64)
    theta = nu * delta_omega * dt

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    alpha = np.empty_like(theta)
    beta = np.empty_like(theta)
    gamma = np.empty_like(theta)

    small = np.abs(theta) < 1.0e-9
    alpha[small] = 0.0
    beta[small] = 2.0 / 3.0
    gamma[small] = 4.0 / 3.0

    regular = ~small
    theta_regular = theta[regular]
    sin_regular = sin_theta[regular]
    cos_regular = cos_theta[regular]
    theta2 = theta_regular * theta_regular
    theta3 = theta2 * theta_regular
    alpha[regular] = (
        theta2 + theta_regular * sin_regular * cos_regular - 2.0 * sin_regular**2
    ) / theta3
    beta[regular] = (
        2.0
        * (theta_regular * (1.0 + cos_regular**2) - 2.0 * sin_regular * cos_regular)
        / theta3
    )
    gamma[regular] = 4.0 * (sin_regular - theta_regular * cos_regular) / theta3

    even_indices = np.arange(0, n_intervals + 1, 2, dtype=np.float64)
    odd_indices = np.arange(1, n_intervals, 2, dtype=np.float64)
    corr_even = corr[::2]
    corr_odd = corr[1:n_intervals:2]

    for inu, theta_value in enumerate(theta):
        cos_even = np.cos(theta_value * even_indices)
        even_sum = np.dot(corr_even, cos_even)
        even_sum -= 0.5 * (
            corr[0] + corr[n_intervals] * np.cos(theta_value * float(n_intervals))
        )

        if odd_indices.size:
            odd_sum = np.dot(corr_odd, np.cos(theta_value * odd_indices))
        else:
            odd_sum = 0.0

        omega = float(inu) * delta_omega
        out[inu] = (
            2.0
            * (
                alpha[inu] * corr[n_intervals] * np.sin(omega * t_max)
                + beta[inu] * even_sum
                + gamma[inu] * odd_sum
            )
            * dt
        )


def _filon_cosine_transform_numba(
    dt: float,
    delta_omega: float,
    n_intervals: int,
    corr: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    if n_intervals % 2 != 0:
        raise ValueError("n_intervals must be even for the Filon transform")
    if corr.shape[0] != n_intervals + 1:
        raise ValueError("corr length must be n_intervals + 1")
    if out.shape[0] != n_intervals + 1:
        raise ValueError("out length must be n_intervals + 1")

    t_max = float(n_intervals) * dt

    for nu in range(n_intervals + 1):
        omega = float(nu) * delta_omega
        theta = omega * dt
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        if abs(theta) < 1.0e-9:
            alpha = 0.0
            beta = 2.0 / 3.0
            gamma = 4.0 / 3.0
        else:
            theta2 = theta * theta
            theta3 = theta2 * theta
            alpha = (
                theta2 + theta * sin_theta * cos_theta - 2.0 * sin_theta**2
            ) / theta3
            beta = (
                2.0
                * (theta * (1.0 + cos_theta**2) - 2.0 * sin_theta * cos_theta)
                / theta3
            )
            gamma = 4.0 * (sin_theta - theta * cos_theta) / theta3

        even_sum = 0.0
        for tau in range(0, n_intervals + 1, 2):
            even_sum += corr[tau] * np.cos(theta * float(tau))
        even_sum -= 0.5 * (
            corr[0] + corr[n_intervals] * np.cos(theta * float(n_intervals))
        )

        odd_sum = 0.0
        for tau in range(1, n_intervals, 2):
            odd_sum += corr[tau] * np.cos(theta * float(tau))

        out[nu] = (
            2.0
            * (
                alpha * corr[n_intervals] * np.sin(omega * t_max)
                + beta * even_sum
                + gamma * odd_sum
            )
            * dt
        )


if NUMBA_AVAILABLE:
    _filon_cosine_transform = njit(cache=True)(_filon_cosine_transform_numba)
else:
    _filon_cosine_transform = _filon_cosine_transform_numpy
