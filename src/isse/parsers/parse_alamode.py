from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..constants import BOHR_TO_ANGSTROM


def read_alamode_evec(
    filename: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read phonon eigenvalues and eigenvectors from an ALAMODE ``.evec`` file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the ALAMODE eigenvector file.

    Returns
    -------
    primitive_cell : numpy.ndarray
        Primitive-cell lattice vectors with shape ``(3, 3)``, expressed in
        angstrom.
    qpoints : numpy.ndarray
        Fractional q-point coordinates with shape ``(nq, 3)``.
    eigenvalues : numpy.ndarray
        Phonon eigenvalues with shape ``(nq, nmodes)``, expressed in
        Rydberg atomic units.
    eigenvectors : numpy.ndarray
        Complex phonon eigenvectors with shape
        ``(nq, nmodes, n_atoms, 3)``.

    Raises
    ------
    ValueError
        If the header is incomplete, the number of modes is invalid, or the
        file contains an inconsistent number or ordering of q-points and modes.
    """

    with open(filename, encoding="utf-8") as handle:
        lines = handle.readlines()

    primitive_cell = None
    nmodes = None
    nq = None

    for i, line in enumerate(lines):
        if line.startswith("# Lattice vectors"):
            primitive_cell = (
                np.array(
                    [
                        [float(value) for value in lines[i + j + 1].split()]
                        for j in range(3)
                    ],
                    dtype=np.float64,
                )
                * BOHR_TO_ANGSTROM
            )
        elif "Number of phonon modes" in line:
            nmodes = int(line.split(":", maxsplit=1)[1])
        elif "Number of k points" in line:
            nq = int(line.split(":", maxsplit=1)[1])

    if primitive_cell is None or nmodes is None or nq is None:
        raise ValueError("Could not read the ALAMODE header.")
    if nmodes % 3 != 0:
        raise ValueError(f"Invalid number of phonon modes: {nmodes}.")

    nat_primitive = nmodes // 3
    qpoints = np.empty((nq, 3), dtype=np.float64)
    eigenvalues = np.empty((nq, nmodes), dtype=np.float64)
    eigenvectors = np.empty((nq, nmodes, nat_primitive, 3), dtype=np.complex128)

    kpoint_re = re.compile(
        r"## kpoint\s+\d+\s*:\s*"
        r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)"
    )
    mode_re = re.compile(r"### mode\s+\d+\s*:\s*([-+0-9.eE]+)")

    iq = -1
    imode = 0
    i = 0

    while i < len(lines):
        k_match = kpoint_re.match(lines[i])
        if k_match:
            iq += 1
            imode = 0
            if iq >= nq:
                raise ValueError("More q-points found than declared in header.")
            qpoints[iq] = np.asarray(k_match.groups(), dtype=np.float64)
            i += 1
            continue

        mode_match = mode_re.match(lines[i])
        if mode_match:
            if iq < 0 or imode >= nmodes:
                raise ValueError("Malformed mode ordering in ALAMODE file.")

            eigenvalues[iq, imode] = float(mode_match.group(1))
            values = np.array(
                [complex(*map(float, lines[i + 1 + j].split())) for j in range(nmodes)],
                dtype=np.complex128,
            )
            eigenvectors[iq, imode] = values.reshape(nat_primitive, 3)
            imode += 1
            i += nmodes + 1
            continue

        i += 1

    if iq + 1 != nq:
        raise ValueError(f"Read {iq + 1} q-points; expected {nq}.")

    return primitive_cell, qpoints, eigenvalues, eigenvectors
