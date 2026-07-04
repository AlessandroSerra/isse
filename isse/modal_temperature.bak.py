#!/usr/bin/env python3
"""Project MD velocities onto ALAMODE phonon eigenvectors.

Configuration is read from a YAML file. The expensive projection is compiled
with Numba and parallelized over independent MD frames. Frames are processed
in batches to limit peak memory usage.

Usage
-----
python modal_temperature_numba.py config.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml
from ase.io import iread, read
from ase.units import kB
from numba import get_num_threads, njit, prange, set_num_threads

from .parsers.parse_alamode import read_alamode_evec

DEFAULT_BATCH_SIZE = 32
DEFAULT_MAPPING_TOLERANCE = 5.0e-2
DEFAULT_TIME_STEP_FS = 1.0
DEFAULT_FRAME_STRIDE = 1
AMU_A2_FS2_TO_EV = 103.64269652680505


@njit(parallel=True, cache=True)
def project_batch_numba(
    velocities: np.ndarray,
    masses: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a batch of Cartesian velocities onto all phonon modes."""
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


def map_atoms_to_primitive(
    atoms,
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


def precompute_coefficients(
    qpoints: np.ndarray,
    eigenvectors: np.ndarray,
    cell_indices: np.ndarray,
    basis_indices: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """Precompute all frame-independent projection coefficients."""
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


def minimum_image_displacements(reference, frame) -> np.ndarray:
    """Return atomic displacement magnitudes under the minimum-image convention."""
    if len(reference) != len(frame):
        raise ValueError("Reference and trajectory frame have different atom counts.")

    displacement = frame.get_positions() - reference.get_positions()
    scaled = displacement @ np.linalg.inv(reference.cell.array)
    scaled -= np.rint(scaled)
    mic = scaled @ reference.cell.array
    return np.linalg.norm(mic, axis=1)


def extract_velocities(frame, velocity_array: str | None) -> np.ndarray:
    """Extract velocities, optionally from a named extxyz array."""
    if velocity_array is not None:
        if velocity_array not in frame.arrays:
            raise KeyError(
                f"Velocity array {velocity_array!r} not found. "
                f"Available arrays: {sorted(frame.arrays)}"
            )
        velocities = np.asarray(frame.arrays[velocity_array], dtype=np.float64)
    else:
        velocities = frame.get_velocities()
        if velocities is None:
            raise ValueError("The ASE frame does not contain velocities.")
        velocities = np.asarray(velocities, dtype=np.float64)

    if velocities.shape != (len(frame), 3):
        raise ValueError(f"Unexpected velocity shape: {velocities.shape}.")
    return velocities


def batched(iterator: Iterator, batch_size: int) -> Iterator[list]:
    """Yield fixed-size lists from an iterator."""
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_config(filename: str | Path) -> dict[str, Any]:
    """Load and minimally validate the YAML configuration."""
    path = Path(filename)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping.")

    required = ("trajectory", "reference", "evec", "basis")
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Missing required YAML entries: {missing}.")

    return config


def resolve_path(value: str | Path, config_dir: Path) -> Path:
    """Resolve relative paths with respect to the YAML directory."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else config_dir / path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project an extxyz MD trajectory onto ALAMODE phonon modes."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help="YAML configuration file (default: config.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    config_dir = config_path.parent

    trajectory_path = resolve_path(config["trajectory"], config_dir)
    reference_path = resolve_path(config["reference"], config_dir)
    evec_path = resolve_path(config["evec"], config_dir)
    output_prefix = resolve_path(config.get("output", "modal_projection"), config_dir)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    batch_size = int(config.get("batch_size", DEFAULT_BATCH_SIZE))
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    threads = config.get("threads", "max")
    if threads not in (None, "max"):
        threads = int(threads)
        if threads < 1:
            raise ValueError("threads must be positive or 'max'.")
        max_threads = os.cpu_count() or threads
        set_num_threads(min(threads, max_threads))

    mapping_tolerance = float(
        config.get("mapping_tolerance", DEFAULT_MAPPING_TOLERANCE)
    )
    velocity_array_value = config.get("velocity_array", "vel")
    velocity_array = (
        None
        if velocity_array_value is None or str(velocity_array_value).lower() == "none"
        else str(velocity_array_value)
    )
    selected_iqs = np.asarray(config.get("selected_iqs", []), dtype=np.int64)
    basis = np.asarray(config["basis"], dtype=np.float64)

    time_step_fs = float(config.get("time_step_fs", DEFAULT_TIME_STEP_FS))
    frame_stride = int(config.get("frame_stride", DEFAULT_FRAME_STRIDE))
    if time_step_fs <= 0.0:
        raise ValueError("time_step_fs must be positive.")
    if frame_stride < 1:
        raise ValueError("frame_stride must be a positive integer.")

    velocity_units = str(config.get("velocity_units", "angstrom/fs")).lower()
    if velocity_units in {"angstrom/fs", "a/fs", "ang/fs"}:
        energy_conversion = AMU_A2_FS2_TO_EV
    elif velocity_units in {"ase", "ase_internal"}:
        energy_conversion = 1.0
    else:
        raise ValueError("velocity_units must be 'angstrom/fs' or 'ase'.")

    reference = read(reference_path, format="extxyz")
    first_frame = read(trajectory_path, index=0, format="extxyz")

    primitive_cell, qpoints, eigenvalues, eigenvectors = read_alamode_evec(evec_path)
    natoms = len(reference)
    nq, nmodes, nbasis_evec, _ = eigenvectors.shape

    if basis.shape != (nbasis_evec, 3):
        raise ValueError(
            f"YAML basis has shape {basis.shape}, while ALAMODE eigenvectors "
            f"require ({nbasis_evec}, 3)."
        )

    ncells = natoms // nbasis_evec
    if len(first_frame) != natoms:
        raise ValueError(
            f"Trajectory has {len(first_frame)} atoms; reference has {natoms}."
        )
    if nq != ncells:
        raise ValueError(
            "Incomplete/incompatible modal basis: "
            f"nq={nq}, but natoms/nbasis={natoms}/{nbasis_evec}={ncells}. "
            f"Expected {ncells} commensurate q-points."
        )
    if nq * nmodes != 3 * natoms:
        raise ValueError(
            f"Mode count mismatch: {nq}*{nmodes}={nq * nmodes}, but 3N={3 * natoms}."
        )
    if selected_iqs.size and (selected_iqs.min() < 0 or selected_iqs.max() >= nq):
        raise IndexError(f"selected_iqs must lie in [0, {nq - 1}].")

    cell_indices, basis_indices, residuals = map_atoms_to_primitive(
        reference,
        primitive_cell,
        basis,
        mapping_tolerance,
    )
    masses = np.asarray(reference.get_masses(), dtype=np.float64)
    coefficients = precompute_coefficients(
        qpoints, eigenvectors, cell_indices, basis_indices, masses
    )
    displacement_norms = minimum_image_displacements(reference, first_frame)

    print(f"Configuration         : {config_path}")
    print(f"Atoms                 : {natoms}")
    print(f"Primitive basis atoms : {nbasis_evec}")
    print(f"Primitive cells       : {ncells}")
    print(f"q-points              : {nq}")
    print(f"Modes                 : {nq * nmodes}")
    print(f"Batch size            : {batch_size}")
    print(f"Numba threads         : {get_num_threads()}")
    print(f"Time step             : {time_step_fs:g} fs")
    print(f"Frame stride          : {frame_stride}")
    print(f"Frame spacing         : {time_step_fs * frame_stride:g} fs")
    print(f"Velocity units        : {velocity_units}")
    print(f"Coefficient memory    : {coefficients.nbytes / 2**20:.1f} MiB")
    print(f"Mapping max residual  : {residuals.max():.6e}")
    print(
        f"Basis populations     : {np.bincount(basis_indices, minlength=nbasis_evec)}"
    )
    print(f"Mean first-frame |dr| : {displacement_norms.mean():.6e} A")
    print(f"Max first-frame |dr|  : {displacement_norms.max():.6e} A")

    sum_qdot2 = np.zeros((nq, nmodes), dtype=np.float64)
    parseval_errors_parts: list[np.ndarray] = []
    selected_temperature_parts: list[np.ndarray] = []
    nframes = 0

    trajectory = iread(trajectory_path, index=":", format="extxyz")

    for ibatch, frames in enumerate(batched(trajectory, batch_size), start=1):
        velocities = np.ascontiguousarray(
            np.stack([extract_velocities(frame, velocity_array) for frame in frames]),
            dtype=np.float64,
        )

        qdot2, atomic_norms = project_batch_numba(velocities, masses, coefficients)
        modal_norms = qdot2.sum(axis=(1, 2))

        errors = np.full_like(atomic_norms, np.nan)
        nonzero = atomic_norms != 0.0
        errors[nonzero] = (
            np.abs(modal_norms[nonzero] - atomic_norms[nonzero]) / atomic_norms[nonzero]
        )

        sum_qdot2 += qdot2.sum(axis=0)
        parseval_errors_parts.append(errors)

        if selected_iqs.size:
            selected_temperature_parts.append(
                qdot2[:, selected_iqs, :] * energy_conversion / kB
            )

        nframes += len(frames)
        print(
            f"Processed {nframes} frames "
            f"(batch {ibatch}, current max Parseval error="
            f"{np.nanmax(errors):.3e})",
            flush=True,
        )

    if nframes == 0:
        raise ValueError("The trajectory contains no frames.")

    parseval_errors = np.concatenate(parseval_errors_parts)
    mean_qdot2 = sum_qdot2 / nframes
    mode_temperatures = mean_qdot2 * energy_conversion / kB

    gamma_index = int(np.argmin(np.linalg.norm(qpoints, axis=1)))
    thermal_mask = np.ones_like(mode_temperatures, dtype=bool)
    thermal_mask[gamma_index, :3] = False

    ndof = 3 * natoms - 3
    reconstructed_temperature = mean_qdot2.sum() * energy_conversion / (ndof * kB)
    mean_thermal_mode_temperature = mode_temperatures[thermal_mask].mean()

    time_fs = np.arange(nframes, dtype=np.float64) * time_step_fs * frame_stride
    time_ps = time_fs / 1000.0

    selected_temperatures = (
        np.concatenate(selected_temperature_parts, axis=0)
        if selected_temperature_parts
        else np.empty((nframes, 0, nmodes), dtype=np.float64)
    )

    np.savez_compressed(
        f"{output_prefix}.npz",
        qpoints=qpoints,
        eigenvalues=eigenvalues,
        mean_qdot2=mean_qdot2,
        mode_temperatures=mode_temperatures,
        parseval_errors=parseval_errors,
        selected_iqs=selected_iqs,
        selected_mode_temperatures=selected_temperatures,
        gamma_index=gamma_index,
        reconstructed_temperature=reconstructed_temperature,
        mean_thermal_mode_temperature=mean_thermal_mode_temperature,
        basis=basis,
        time_fs=time_fs,
        time_ps=time_ps,
        time_step_fs=time_step_fs,
        frame_stride=frame_stride,
        velocity_units=velocity_units,
        energy_conversion=energy_conversion,
    )

    summary = (
        f"Frames: {nframes}\n"
        f"Gamma index: {gamma_index}\n"
        f"Time step [fs]: {time_step_fs:.12g}\n"
        f"Frame stride: {frame_stride}\n"
        f"Frame spacing [fs]: {time_step_fs * frame_stride:.12g}\n"
        f"Total sampled time [ps]: {time_ps[-1]:.12g}\n"
        f"Velocity units: {velocity_units}\n"
        f"Gamma acoustic temperatures [K]: "
        f"{mode_temperatures[gamma_index, :3]}\n"
        f"Mean thermal modal temperature [K]: "
        f"{mean_thermal_mode_temperature:.12g}\n"
        f"Temperature reconstructed from modes [K]: "
        f"{reconstructed_temperature:.12g}\n"
        f"Parseval mean relative error: "
        f"{np.nanmean(parseval_errors):.12e}\n"
        f"Parseval max relative error: "
        f"{np.nanmax(parseval_errors):.12e}\n"
        f"Parseval min relative error: "
        f"{np.nanmin(parseval_errors):.12e}\n"
    )
    Path(f"{output_prefix}_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    main()
