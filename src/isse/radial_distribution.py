from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from .helpers.periodic import minimum_image_distances
from .structures import Atoms, Trajectory


def calculate_rdf(
    trajectory: Trajectory | Atoms,
    r_max: float,
    n_bins: int,
    batch_size: int = 100,
    partial: bool = False,
    species_pairs: Sequence[tuple[str, str]] | None = None,
) -> dict[str, NDArray[np.float64] | dict[tuple[str, str], NDArray[np.float64]]]:
    """
    Calculate total and, optionally, partial radial distribution functions.

    Parameters
    ----------
    trajectory : Trajectory or Atoms
        Lazy trajectory yielding one ``Atoms`` object per frame, or a single
        ``Atoms`` configuration.
    r_max : float
        Maximum distance included in the RDF histogram.
    n_bins : int
        Number of bins for the RDF histogram.
    batch_size : int, optional
        Maximum number of trajectory frames processed in each batch.
    partial : bool, optional
        If ``True``, calculate one partial RDF for each species pair.
    species_pairs : sequence of tuple[str, str], optional
        Species pairs for which partial RDFs are calculated. If provided,
        partial RDFs are enabled even when ``partial`` is ``False``. Pair order
        is ignored, so ``("Si", "O")`` and ``("O", "Si")`` are equivalent.

    Returns
    -------
    dict
        Dictionary containing:

        - ``"r"``: bin centers, shape ``(nbins,)``;
        - ``"g_r"``: total RDF, shape ``(nbins,)`` for ``Atoms`` input or
          ``(nframes, nbins)`` for ``Trajectory`` input;
        - ``"partial_g_r"``: optional partial RDFs keyed by species pair.
    """
    if r_max <= 0.0:
        raise ValueError(f"r_max must be positive, found {r_max}.")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, found {batch_size}.")

    if n_bins < 1:
        raise ValueError(f"n_bins must be positive, found {n_bins}.")

    dr = r_max / n_bins

    is_single_frame = isinstance(trajectory, Atoms)
    count_batches: list[NDArray[np.float64]] = []
    partial_count_batches: list[NDArray[np.float64]] = []
    pair_keys: list[tuple[str, str]] = []
    pair_index: NDArray[np.int64] | None = None
    species_ids: NDArray[np.int64] | None = None
    species_counts: dict[str, int] = {}
    volumes: list[float] = []
    nframes = 0
    natoms: int | None = None
    reference_symbols: tuple[str, ...] | None = None
    compute_partial = partial or species_pairs is not None

    for positions, cells, symbols in _iter_position_batches(
        trajectory,
        batch_size=batch_size,
    ):
        if natoms is None:
            natoms = positions.shape[1]
        elif positions.shape[1] != natoms:
            raise ValueError(
                "All trajectory frames must contain the same number of atoms: "
                f"expected {natoms}, found {positions.shape[1]}."
            )

        if reference_symbols is None:
            reference_symbols = symbols
        elif symbols != reference_symbols:
            raise ValueError("All trajectory frames must contain the same symbols.")

        if compute_partial and species_ids is None:
            species_ids, species_counts, pair_keys, pair_index = _prepare_partial_rdf(
                symbols,
                species_pairs,
            )

        inverse_cells = np.linalg.inv(cells)
        if NUMBA_AVAILABLE:
            batch_counts = _histogram_rdf_numba(
                positions,
                cells,
                inverse_cells,
                r_max,
                dr,
                n_bins,
            )
        else:
            batch_counts = _histogram_rdf_numpy(
                positions,
                cells,
                r_max,
                dr,
                n_bins,
            )

        count_batches.append(batch_counts)

        if compute_partial:
            if species_ids is None or pair_index is None:
                raise RuntimeError("Partial RDF metadata was not initialized.")
            if NUMBA_AVAILABLE:
                batch_partial_counts = _histogram_partial_rdf_numba(
                    positions,
                    cells,
                    inverse_cells,
                    species_ids,
                    pair_index,
                    r_max,
                    dr,
                    n_bins,
                    len(pair_keys),
                )
            else:
                batch_partial_counts = _histogram_partial_rdf_numpy(
                    positions,
                    cells,
                    species_ids,
                    pair_index,
                    r_max,
                    dr,
                    n_bins,
                    len(pair_keys),
                )
            partial_count_batches.append(batch_partial_counts)
        volumes.extend(np.abs(np.linalg.det(cells)).tolist())
        nframes += positions.shape[0]

    if natoms is None or nframes == 0:
        raise ValueError("No trajectory frames were read.")

    counts = np.concatenate(count_batches, axis=0)
    volumes_array = np.asarray(volumes, dtype=np.float64)

    r, g_r = _normalize_rdf(
        counts=counts,
        natoms=natoms,
        volumes=volumes_array,
        dr=dr,
    )

    if is_single_frame:
        g_r = g_r[0]

    results: dict[
        str,
        NDArray[np.float64] | dict[tuple[str, str], NDArray[np.float64]],
    ] = {
        "r": r,
        "g_r": g_r,
    }

    if compute_partial:
        if not partial_count_batches:
            raise ValueError("No partial RDF species pairs were available.")
        partial_counts = np.concatenate(partial_count_batches, axis=0)
        partial_g_r = _normalize_partial_rdf(
            counts=partial_counts,
            pair_keys=pair_keys,
            species_counts=species_counts,
            volumes=volumes_array,
            dr=dr,
        )
        if is_single_frame:
            partial_g_r = partial_g_r[0]
        results["partial_g_r"] = {
            pair: partial_g_r[..., ipair, :]
            for ipair, pair in enumerate(pair_keys)
        }

    return results


def _iter_position_batches(
    trajectory: Trajectory | Atoms,
    batch_size: int,
) -> Iterator[tuple[NDArray[np.float64], NDArray[np.float64], tuple[str, ...]]]:
    """
    Yield batches of positions and cells from a trajectory or single frame.

    Yields
    ------
    positions : numpy.ndarray
        Cartesian positions with shape ``(nframes_batch, natoms, 3)``.
    cells : numpy.ndarray
        Cell matrices with shape ``(nframes_batch, 3, 3)``.
    """
    position_batch: list[NDArray[np.float64]] = []
    cell_batch: list[NDArray[np.float64]] = []
    symbols: tuple[str, ...] | None = None
    natoms: int | None = None
    frames = (trajectory,) if isinstance(trajectory, Atoms) else trajectory

    for iframe, atoms in enumerate(frames):
        positions = atoms.positions
        cell = atoms.cell

        if positions.shape[-1] != 3 or positions.ndim != 2:
            raise ValueError(
                f"Frame {iframe} positions must have shape (n_atoms, 3), "
                f"found {positions.shape}."
            )

        if cell.shape != (3, 3):
            raise ValueError(
                f"Frame {iframe} cell must have shape (3, 3), found {cell.shape}."
            )

        frame_symbols = tuple(atoms.symbols)
        if natoms is None:
            natoms = positions.shape[0]
            symbols = frame_symbols
        elif positions.shape[0] != natoms:
            raise ValueError(
                "All trajectory frames must contain the same number of atoms: "
                f"expected {natoms}, found {positions.shape[0]} in frame {iframe}."
            )
        elif frame_symbols != symbols:
            raise ValueError("All trajectory frames must contain the same symbols.")

        position_batch.append(positions)
        cell_batch.append(cell)

        if len(position_batch) == batch_size:
            if symbols is None:
                raise RuntimeError(
                    "Position batch is not empty but symbols are missing."
                )
            yield (
                np.ascontiguousarray(position_batch, dtype=np.float64),
                np.ascontiguousarray(cell_batch, dtype=np.float64),
                symbols,
            )
            position_batch.clear()
            cell_batch.clear()

    if position_batch:
        if symbols is None:
            raise RuntimeError("Position batch is not empty but symbols are missing.")
        yield (
            np.ascontiguousarray(position_batch, dtype=np.float64),
            np.ascontiguousarray(cell_batch, dtype=np.float64),
            symbols,
        )


def _prepare_partial_rdf(
    symbols: tuple[str, ...],
    species_pairs: Sequence[tuple[str, str]] | None,
) -> tuple[NDArray[np.int64], dict[str, int], list[tuple[str, str]], NDArray[np.int64]]:
    """Prepare species ids and pair lookup tables for partial RDFs."""
    species = sorted(set(symbols))
    species_to_id = {symbol: ispecies for ispecies, symbol in enumerate(species)}
    species_counts = {symbol: symbols.count(symbol) for symbol in species}
    species_ids = np.asarray(
        [species_to_id[symbol] for symbol in symbols],
        dtype=np.int64,
    )

    if species_pairs is None:
        pair_keys = [
            (species[i], species[j])
            for i in range(len(species))
            for j in range(i, len(species))
        ]
    else:
        pair_keys = []
        seen: set[tuple[str, str]] = set()
        for pair in species_pairs:
            if len(pair) != 2:
                raise ValueError(f"Invalid species pair {pair!r}.")
            first, second = pair
            if first not in species_to_id or second not in species_to_id:
                raise ValueError(
                    f"Species pair {pair!r} is not present in the trajectory. "
                    f"Available species are {species}."
                )
            if species_to_id[first] <= species_to_id[second]:
                key = (first, second)
            else:
                key = (second, first)
            if key not in seen:
                seen.add(key)
                pair_keys.append(key)

    pair_index = -np.ones((len(species), len(species)), dtype=np.int64)
    for ipair, (first, second) in enumerate(pair_keys):
        i = species_to_id[first]
        j = species_to_id[second]
        pair_index[i, j] = ipair
        pair_index[j, i] = ipair

    return species_ids, species_counts, pair_keys, pair_index


def _histogram_rdf_numpy(
    positions: NDArray[np.float64],
    cells: NDArray[np.float64],
    r_max: float,
    dr: float,
    nbins: int,
) -> NDArray[np.float64]:
    """
    Accumulate RDF pair counts using a readable NumPy fallback.

    Pair distances are computed with the minimum image convention. Each pair
    ``i < j`` contributes ``2`` counts because atom ``i`` sees atom ``j`` and
    atom ``j`` sees atom ``i``.
    """
    nframes, natoms, _ = positions.shape
    counts = np.zeros((nframes, nbins), dtype=np.float64)

    for iframe in range(nframes):
        frame_positions = positions[iframe]
        cell = cells[iframe]
        frame_counts = counts[iframe]

        for iatom in range(natoms - 1):
            displacements = frame_positions[iatom + 1 :] - frame_positions[iatom]
            distances = minimum_image_distances(displacements, cell)
            selected = distances < r_max
            bin_indices = np.floor(distances[selected] / dr).astype(np.int64)
            frame_counts += 2.0 * np.bincount(bin_indices, minlength=nbins)[:nbins]

    return counts


def _histogram_partial_rdf_numpy(
    positions: NDArray[np.float64],
    cells: NDArray[np.float64],
    species_ids: NDArray[np.int64],
    pair_index: NDArray[np.int64],
    r_max: float,
    dr: float,
    nbins: int,
    npairs: int,
) -> NDArray[np.float64]:
    """
    Accumulate partial RDF pair counts using a readable NumPy fallback.
    """
    nframes, natoms, _ = positions.shape
    counts = np.zeros((nframes, npairs, nbins), dtype=np.float64)

    for iframe in range(nframes):
        frame_positions = positions[iframe]
        cell = cells[iframe]
        frame_counts = counts[iframe]

        for iatom in range(natoms - 1):
            displacements = frame_positions[iatom + 1 :] - frame_positions[iatom]
            distances = minimum_image_distances(displacements, cell)
            i_species = species_ids[iatom]

            for joffset, distance in enumerate(distances):
                if distance >= r_max:
                    continue
                jatom = iatom + 1 + joffset
                ipair = pair_index[i_species, species_ids[jatom]]
                if ipair < 0:
                    continue
                ibin = int(distance / dr)
                frame_counts[ipair, ibin] += 2.0

    return counts


if NUMBA_AVAILABLE:

    @njit(cache=True, parallel=True)
    def _histogram_rdf_numba(
        positions: NDArray[np.float64],
        cells: NDArray[np.float64],
        inverse_cells: NDArray[np.float64],
        r_max: float,
        dr: float,
        nbins: int,
    ) -> NDArray[np.float64]:
        """
        Accumulate RDF pair counts using a Numba backend parallel over frames.
        """
        nframes, natoms, _ = positions.shape
        counts_by_frame = np.zeros((nframes, nbins), dtype=np.float64)

        for iframe in prange(nframes):
            cell = cells[iframe]
            inverse_cell = inverse_cells[iframe]
            frame_counts = counts_by_frame[iframe]

            for iatom in range(natoms - 1):
                xi = positions[iframe, iatom, 0]
                yi = positions[iframe, iatom, 1]
                zi = positions[iframe, iatom, 2]

                for jatom in range(iatom + 1, natoms):
                    dx = positions[iframe, jatom, 0] - xi
                    dy = positions[iframe, jatom, 1] - yi
                    dz = positions[iframe, jatom, 2] - zi

                    sx = (
                        dx * inverse_cell[0, 0]
                        + dy * inverse_cell[1, 0]
                        + dz * inverse_cell[2, 0]
                    )
                    sy = (
                        dx * inverse_cell[0, 1]
                        + dy * inverse_cell[1, 1]
                        + dz * inverse_cell[2, 1]
                    )
                    sz = (
                        dx * inverse_cell[0, 2]
                        + dy * inverse_cell[1, 2]
                        + dz * inverse_cell[2, 2]
                    )

                    sx -= np.rint(sx)
                    sy -= np.rint(sy)
                    sz -= np.rint(sz)

                    dx = sx * cell[0, 0] + sy * cell[1, 0] + sz * cell[2, 0]
                    dy = sx * cell[0, 1] + sy * cell[1, 1] + sz * cell[2, 1]
                    dz = sx * cell[0, 2] + sy * cell[1, 2] + sz * cell[2, 2]

                    distance = np.sqrt(dx * dx + dy * dy + dz * dz)

                    if distance < r_max:
                        ibin = int(distance / dr)
                        frame_counts[ibin] += 2.0

        return counts_by_frame


    @njit(cache=True, parallel=True)
    def _histogram_partial_rdf_numba(
        positions: NDArray[np.float64],
        cells: NDArray[np.float64],
        inverse_cells: NDArray[np.float64],
        species_ids: NDArray[np.int64],
        pair_index: NDArray[np.int64],
        r_max: float,
        dr: float,
        nbins: int,
        npairs: int,
    ) -> NDArray[np.float64]:
        """
        Accumulate partial RDF pair counts using a Numba backend.
        """
        nframes, natoms, _ = positions.shape
        counts_by_frame = np.zeros((nframes, npairs, nbins), dtype=np.float64)

        for iframe in prange(nframes):
            cell = cells[iframe]
            inverse_cell = inverse_cells[iframe]
            frame_counts = counts_by_frame[iframe]

            for iatom in range(natoms - 1):
                xi = positions[iframe, iatom, 0]
                yi = positions[iframe, iatom, 1]
                zi = positions[iframe, iatom, 2]
                i_species = species_ids[iatom]

                for jatom in range(iatom + 1, natoms):
                    ipair = pair_index[i_species, species_ids[jatom]]
                    if ipair < 0:
                        continue

                    dx = positions[iframe, jatom, 0] - xi
                    dy = positions[iframe, jatom, 1] - yi
                    dz = positions[iframe, jatom, 2] - zi

                    sx = (
                        dx * inverse_cell[0, 0]
                        + dy * inverse_cell[1, 0]
                        + dz * inverse_cell[2, 0]
                    )
                    sy = (
                        dx * inverse_cell[0, 1]
                        + dy * inverse_cell[1, 1]
                        + dz * inverse_cell[2, 1]
                    )
                    sz = (
                        dx * inverse_cell[0, 2]
                        + dy * inverse_cell[1, 2]
                        + dz * inverse_cell[2, 2]
                    )

                    sx -= np.rint(sx)
                    sy -= np.rint(sy)
                    sz -= np.rint(sz)

                    dx = sx * cell[0, 0] + sy * cell[1, 0] + sz * cell[2, 0]
                    dy = sx * cell[0, 1] + sy * cell[1, 1] + sz * cell[2, 1]
                    dz = sx * cell[0, 2] + sy * cell[1, 2] + sz * cell[2, 2]

                    distance = np.sqrt(dx * dx + dy * dy + dz * dz)

                    if distance < r_max:
                        ibin = int(distance / dr)
                        frame_counts[ipair, ibin] += 2.0

        return counts_by_frame


def _normalize_rdf(
    counts: NDArray[np.float64],
    natoms: int,
    volumes: NDArray[np.float64],
    dr: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Normalize per-frame RDF pair counts into ``g(r)``.
    """
    nbins = counts.shape[1]
    edges, shell_volumes = _rdf_edges_and_shell_volumes(nbins, dr)
    r = 0.5 * (edges[:-1] + edges[1:])
    densities = natoms / volumes

    normalization = natoms * densities[:, np.newaxis] * shell_volumes[np.newaxis, :]
    g_r = np.divide(
        counts,
        normalization,
        out=np.zeros_like(counts, dtype=np.float64),
        where=normalization > 0.0,
    )

    return r, np.asarray(g_r, dtype=np.float64)


def _normalize_partial_rdf(
    counts: NDArray[np.float64],
    pair_keys: Sequence[tuple[str, str]],
    species_counts: dict[str, int],
    volumes: NDArray[np.float64],
    dr: float,
) -> NDArray[np.float64]:
    """
    Normalize per-frame partial RDF counts into species-resolved ``g_ab(r)``.

    Counts are accumulated as ordered neighbor counts: each unordered pair
    contributes ``2`` counts. For cross pairs this is normalized by the sum of
    the two directional ideal-gas counts, ``N_a rho_b + N_b rho_a``.
    """
    nframes, npairs, nbins = counts.shape
    _, shell_volumes = _rdf_edges_and_shell_volumes(nbins, dr)
    g_r = np.zeros((nframes, npairs, nbins), dtype=np.float64)

    for ipair, (first, second) in enumerate(pair_keys):
        n_first = species_counts[first]
        n_second = species_counts[second]
        if first == second:
            prefactors = n_first * (n_first / volumes)
        else:
            prefactors = (
                n_first * (n_second / volumes)
                + n_second * (n_first / volumes)
            )
        normalization = prefactors[:, np.newaxis] * shell_volumes[np.newaxis, :]
        g_r[:, ipair, :] = np.divide(
            counts[:, ipair, :],
            normalization,
            out=np.zeros((nframes, nbins), dtype=np.float64),
            where=normalization > 0.0,
        )

    return g_r


def _rdf_edges_and_shell_volumes(
    nbins: int,
    dr: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return RDF bin edges and spherical shell volumes."""
    edges = np.arange(nbins + 1, dtype=np.float64) * dr
    shell_volumes = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    return edges, shell_volumes
