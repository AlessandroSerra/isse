# ISSE

**Ichnusa Solid State Environment** is a Python toolkit for solid-state physics and atomistic simulation workflows.

The project provides lightweight atomistic data structures, lazy trajectory I/O, format conversion utilities, periodic-boundary helpers, radial distribution functions, and phonon-mode velocity projection workflows.

> Status: early development. APIs are usable but not yet guaranteed stable.

## Features

- `Atoms`: a compact container for one atomistic configuration.
- `Trajectory`: lazy, file-backed access to single-file or multi-file trajectories.
- Native readers and writers for selected VASP, LAMMPS, and GPUMD/extended-XYZ files.
- Explicit unit handling for LAMMPS `metal` and `real` unit styles.
- Dependency-light conversion utilities without ASE.
- Phonon modal temperature workflow using ALAMODE eigenvector files.
- Radial distribution function calculation with optional Numba acceleration.
- Velocity autocorrelation and vibrational density of states utilities.
- Periodic-boundary utilities for wrapping, unwrapping, and minimum-image distances.

## Installation

ISSE requires Python 3.11 or newer.

From a local checkout:

```bash
pip install .
```

or, with `uv`:

```bash
uv sync
```

Runtime dependencies are:

- `numpy`
- `spglib`

Optional performance dependency:

- `numba`, installable with `pip install .[performance]` or the equivalent `uv` extra.

If `numba` is not installed, ISSE falls back to NumPy implementations where available.

## Internal unit convention

ISSE converts parsed data to a consistent internal unit system:

| Quantity | Internal unit |
| --- | --- |
| Cell vectors | Å |
| Positions / unwrapped positions | Å |
| Velocities | Å/fs |
| Masses | amu |
| Forces | eV/Å |
| Energies used internally | eV |
| Time metadata, when interpreted | fs |
| Temperature | K |

This convention is important when constructing `Atoms` objects manually or when passing data to writers.

## Supported file formats

| Format name | Read | Write | Object | Notes |
| --- | --- | --- | --- | --- |
| `poscar`, `vasp` | yes | yes | `Atoms` | POSCAR-like single configuration. Positive scaling factors only. Direct and Cartesian positions supported. |
| `lammps-data` | yes | yes | `Atoms` | `atom_style atomic`; optional `Masses` and `Velocities`; orthogonal or restricted-triclinic cells. |
| `lammps-dump`, `lammpstrj` | yes | yes | `Trajectory` | Lazy multi-frame dump reader; supports `x/y/z`, `xs/ys/zs`, `xu/yu/zu`, velocities, forces, masses, ids, types. |
| `gpumd-dump`, `extxyz` | yes | yes | `Trajectory` | GPUMD/extended-XYZ style with `Lattice` and `Properties`. |

Aliases accepted by the converter include `data`, `dump`, `lammps_data`, `lammps_dump`, `xyz`, and `gpumd`.

## LAMMPS units

LAMMPS files must be read or written with an explicit unit style: `"metal"` or `"real"`. ISSE intentionally avoids silent assumptions.

### `units="metal"`

| Quantity | LAMMPS unit | ISSE internal conversion |
| --- | --- | --- |
| Positions/cell | Å | unchanged |
| Velocities | Å/ps | divided by 1000 on read; multiplied by 1000 on write |
| Forces | eV/Å | unchanged |
| Masses | amu | unchanged |

### `units="real"`

| Quantity | LAMMPS unit | ISSE internal conversion |
| --- | --- | --- |
| Positions/cell | Å | unchanged |
| Velocities | Å/fs | unchanged |
| Forces | kcal/mol/Å | multiplied by `0.0433641` on read; divided by it on write |
| Masses | amu | unchanged |

For LAMMPS data files, the parser first tries to infer the unit style from the first line, for example an ISSE-written header such as `ISSE LAMMPS data file (units metal)`. If it cannot infer the style, pass `units` explicitly.

LAMMPS data files do not store element symbols directly. Pass `symbols=[...]` in atom-type order when symbols cannot be inferred from masses or when precise labels are required:

```python
from isse.io.parse_lammps import parse_lammps

atoms = parse_lammps(
    "structure.data",
    format="data",
    units="metal",
    symbols=["Si", "O"],  # type 1 -> Si, type 2 -> O
)
```

## Basic usage

### Read a POSCAR

```python
from isse.io.parse_vasp import parse_poscar

atoms = parse_poscar("POSCAR")
print(len(atoms))
print(atoms.cell)
print(atoms.positions)
```

### Read a lazy LAMMPS trajectory

```python
from isse.io.parse_lammps import parse_lammps

trajectory = parse_lammps(
    "dump.lammpstrj",
    format="dump",
    units="metal",
    symbols=["Si"],
)

print(len(trajectory))      # number of frames
frame0 = trajectory[0]      # only this frame is read from disk
for frame in trajectory[:10]:
    print(frame.positions.shape)
```

### Concatenate trajectory segments lazily

Trajectory segments read with the same parser can be concatenated without
loading frames into memory:

```python
from isse.io.parse_gpumddump import parse_gpumd_dump

trajectory = sum([
    parse_gpumd_dump("traj2ps.xyz"),
    parse_gpumd_dump("traj10ps.xyz"),
    parse_gpumd_dump("traj30ps.xyz"),
])

print(len(trajectory))
print(trajectory.paths)     # all backing files
frame = trajectory[-1]      # only this frame is read
```

### Convert files

The conversion API uses ISSE's native readers and writers:

```python
from isse.convert import convert

convert(
    "dump.lammpstrj",
    "trajectory.xyz",
    infile_type="lammps-dump",
    outfile_type="gpumd-dump",
    input_units="metal",
    symbols=["Si"],
)
```

Select one frame when writing a single-frame output from a trajectory:

```python
convert(
    "dump.lammpstrj",
    "frame10.POSCAR",
    infile_type="lammps-dump",
    outfile_type="poscar",
    input_units="metal",
    symbols=["Si"],
    frame=10,
    fractional=True,
)
```

Replicate before writing:

```python
convert(
    "POSCAR",
    "supercell.data",
    infile_type="poscar",
    outfile_type="lammps-data",
    output_units="metal",
    replicate=(2, 2, 2),
)
```

### Write files directly

```python
from isse.io.write_vasp import write_poscar
from isse.io.write_lammps import write_lammps_dump

write_poscar("POSCAR.out", atoms, direct=True)
write_lammps_dump("out.lammpstrj", trajectory, units="metal", fractional=False)
```

## Data model

### `Atoms`

`Atoms` represents one atomistic configuration:

```python
from isse.structures import Atoms
```

Main fields:

- `symbols`: list of chemical symbols.
- `cell`: `(3, 3)` array, lattice vectors by row.
- `positions`: `(n_atoms, 3)` Cartesian positions in Å.
- `unwrapped_positions`: optional `(n_atoms, 3)` array in Å.
- `velocities`: optional `(n_atoms, 3)` array in Å/fs.
- `masses`: optional `(n_atoms,)` array in amu.
- `forces`: optional `(n_atoms, 3)` array in eV/Å.
- `arrays`: extra per-atom arrays such as `id`, `type`, or `groups`.
- `info`: frame/configuration metadata such as `timestep` or POSCAR header.

Cell convention:

```text
cartesian_position = fractional_position @ cell
```

### `Trajectory`

`Trajectory` provides lazy sequence semantics over frames stored in a file:

- `trajectory[i]` reads one frame.
- `trajectory[start:stop]` returns another lazy trajectory view.
- Iteration reads frames on demand.
- Memory usage scales with the number of frame offsets, not with the full trajectory size.

## Radial distribution function

```python
from isse.radial_distribution import calculate_rdf

rdf = calculate_rdf(trajectory, r_max=8.0, n_bins=400)
print(rdf["r"])       # shape (n_bins,)
print(rdf["g_r"])     # shape (n_frames, n_bins)

partial_rdf = calculate_rdf(trajectory, r_max=8.0, n_bins=400, partial=True)
print(partial_rdf["partial_g_r"][("O", "Si")])  # shape (n_frames, n_bins)
```

The RDF implementation assumes periodic boundary conditions and returns one RDF per frame for trajectory inputs. Passing a single `Atoms` object returns one-dimensional arrays for that single configuration. If `numba` is installed, the accelerated backend is used automatically; otherwise ISSE falls back to a NumPy implementation.

## Phonon modal temperatures

```python
from isse.io.parse_gpumddump import parse_gpumd_dump
from isse.io.parse_vasp import parse_poscar
from isse.phonon_temperatures import calculate_temperature

trajectory = parse_gpumd_dump("production.xyz")
reference = parse_poscar("POSCAR")

results = calculate_temperature(
    trajectory=trajectory,
    reference_atoms=reference,
    evec_filepath="modes.evec",
    batch_size=100,
)

print(results["qpoints"])
print(results["mode_temperatures"])
print(results["mean_mode_temperature"])
```

Requirements for this workflow:

- The trajectory frames must include velocities in Å/fs.
- `reference_atoms.masses` must be available.
- The ALAMODE `.evec` file is read by `isse.io.parse_alamode.read_alamode_evec`.
- A Parseval consistency check is performed by default; pass `parseval_tolerance=None` to disable it.

## Development notes

The public API is currently module-based; import from the specific modules shown above. The top-level `isse` package does not yet re-export a stable aggregate API.

Recommended checks while developing:

```bash
python -m compileall src/isse
```

If tests are available in your checkout:

```bash
python -m pytest
```

## License

See `LICENSE`.
