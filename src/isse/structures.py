from __future__ import annotations

import functools
from bisect import bisect_right
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import overload

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class Atoms:
    """
    Store an atomistic configuration.

    Parameters
    ----------
    symbols : list[str]
        Chemical symbols of the atoms.
    cell : NDArray[np.float64]
        Simulation cell with shape ``(3, 3)``.
    positions : NDArray[np.float64]
        Atomic positions with shape ``(n_atoms, 3)``.
    unwrapped_positions : NDArray[np.float64] or None, optional
        Atomic unwrapped positions with shape ``(n_atoms, 3)``.
    velocities : NDArray[np.float64] or None, optional
        Atomic velocities with shape ``(n_atoms, 3)``.
    masses : NDArray[np.float64] or None, optional
        Atomic masses with shape ``(n_atoms,)``.
    forces : NDArray[np.float64] or None, optional
        Atomic forces with shape ``(n_atoms, 3)``.
    arrays : dict[str, np.ndarray], optional
        Additional per-atom arrays.
    info : dict[str, str], optional
        Additional metadata.

    Examples
    --------
    Create a configuration containing two silicon atoms:

    >>> symbols = ["Si", "Si"]
    >>> cell = np.eye(3) * 5.43
    >>> positions = np.array(
    ...     [
    ...         [0.0, 0.0, 0.0],
    ...         [1.3575, 1.3575, 1.3575],
    ...     ],
    ...     dtype=np.float64,
    ... )
    >>> atoms = Atoms(
    ...     symbols=symbols,
    ...     cell=cell,
    ...     positions=positions,
    ... )
    >>> len(atoms)
    2
    >>> atoms.has_velocities
    False
    """

    symbols: list[str]
    cell: NDArray[np.float64]
    positions: NDArray[np.float64]
    unwrapped_positions: NDArray[np.float64] | None = None
    velocities: NDArray[np.float64] | None = None
    masses: NDArray[np.float64] | None = None
    forces: NDArray[np.float64] | None = None
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate array shapes."""
        n_atoms = len(self.symbols)

        if self.cell.shape != (3, 3):
            raise ValueError("cell must have shape (3, 3)")

        if self.positions.shape != (n_atoms, 3):
            raise ValueError("positions must have shape (n_atoms, 3)")

        if self.unwrapped_positions is not None and self.unwrapped_positions.shape != (
            n_atoms,
            3,
        ):
            raise ValueError("unwrapped_positions must have shape (n_atoms, 3)")

        if self.masses is not None and self.masses.shape != (n_atoms,):
            raise ValueError("masses must have shape (n_atoms,)")

        if self.velocities is not None and self.velocities.shape != (n_atoms, 3):
            raise ValueError("velocities must have shape (n_atoms, 3)")

        if self.forces is not None and self.forces.shape != (n_atoms, 3):
            raise ValueError("forces must have shape (n_atoms, 3)")

    @property
    def has_unwrapped_positions(self) -> bool:
        """Return whether unwrapped positions are available."""
        return self.unwrapped_positions is not None

    @property
    def has_velocities(self) -> bool:
        """Return whether velocities are available."""
        return self.velocities is not None

    @property
    def has_masses(self) -> bool:
        """Return whether masses are available."""
        return self.masses is not None

    @property
    def has_forces(self) -> bool:
        """Return whether forces are available."""
        return self.forces is not None

    def __len__(self) -> int:
        """Return the number of atoms."""
        return len(self.symbols)

    def __repr__(self) -> str:
        """Return a compact string representation."""
        return (
            f"{type(self).__name__}("
            f"n_atoms={len(self)}, "
            f"unwrapped_positions={self.has_unwrapped_positions}, "
            f"velocities={self.has_velocities}, "
            f"masses={self.has_masses}, "
            f"forces={self.has_forces}"
            ")"
        )

    def __str__(self) -> str:
        return self.__repr__()


FrameReader = Callable[[Path, int], Atoms]


def _same_reader(left: FrameReader, right: FrameReader) -> bool:
    """Return whether two frame readers can be treated as equivalent."""
    if left is right:
        return True

    if isinstance(left, functools.partial) and isinstance(right, functools.partial):
        return (
            left.func is right.func
            and left.args == right.args
            and (left.keywords or {}) == (right.keywords or {})
        )

    return False


class Trajectory(Sequence[Atoms]):
    """
    Provide lazy access to frames in one or more atomistic trajectory files.

    Frames are read from disk only when accessed. The trajectory stores byte
    offsets rather than one :class:`Atoms` instance per timestep. Multiple
    trajectories can be concatenated lazily with ``+`` when they use the same
    frame reader.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the trajectory file.
    offsets : Sequence[int]
        Byte offsets marking the beginning of each frame.
    reader : Callable[[pathlib.Path, int], Atoms]
        Function used to read one frame from the trajectory file. The function
        receives the file path and the byte offset of the frame.

    Notes
    -----
    Accessing a frame creates a new :class:`Atoms` instance. Frames that are
    never accessed are never loaded into memory.

    The memory required by the trajectory object is proportional to the
    number of stored frame offsets rather than to the total size of the
    trajectory data.

    Examples
    --------
    >>> trajectory = Trajectory(
    ...     path="trajectory.xyz",
    ...     offsets=offsets,
    ...     reader=read_xyz_frame,
    ... )
    >>> len(trajectory)
    1000
    >>> frame = trajectory[10]
    >>> isinstance(frame, Atoms)
    True
    """

    __slots__ = ("_paths", "_offsets", "_reader", "_cumulative")

    def __init__(
        self,
        path: str | Path,
        offsets: Sequence[int],
        reader: FrameReader,
    ) -> None:
        self._init_from_parts(
            paths=(Path(path),),
            offsets=(tuple(offsets),),
            reader=reader,
        )

    def _init_from_parts(
        self,
        paths: Sequence[Path],
        offsets: Sequence[Sequence[int]],
        reader: FrameReader,
    ) -> None:
        """Initialize from one or more ``(path, offsets)`` sources."""
        if len(paths) != len(offsets):
            raise ValueError("paths and offsets must have the same length")

        self._paths = tuple(Path(path) for path in paths)
        self._offsets = tuple(tuple(frame_offsets) for frame_offsets in offsets)
        self._reader = reader

        for path in self._paths:
            if not path.is_file():
                raise FileNotFoundError(path)

        if any(offset < 0 for frame_offsets in self._offsets for offset in frame_offsets):
            raise ValueError("frame offsets must be non-negative")

        cumulative: list[int] = []
        total = 0
        for frame_offsets in self._offsets:
            total += len(frame_offsets)
            cumulative.append(total)
        self._cumulative = tuple(cumulative)

    @classmethod
    def _from_parts(
        cls,
        paths: Sequence[Path],
        offsets: Sequence[Sequence[int]],
        reader: FrameReader,
    ) -> Trajectory:
        """Build a trajectory from already known sources."""
        trajectory = cls.__new__(cls)
        trajectory._init_from_parts(paths, offsets, reader)
        return trajectory

    @property
    def path(self) -> Path:
        """Return the path to the trajectory file.

        Multi-file trajectories have no single path; use :attr:`paths` instead.
        """
        if len(self._paths) != 1:
            raise AttributeError("multi-file trajectories have multiple paths; use paths")
        return self._paths[0]

    @property
    def paths(self) -> tuple[Path, ...]:
        """Return all paths backing the trajectory."""
        return self._paths

    def __len__(self) -> int:
        """Return the number of frames."""
        return self._cumulative[-1] if self._cumulative else 0

    def _locate(self, index: int) -> tuple[int, int]:
        """Return the source index and local frame index for ``index``."""
        source_index = bisect_right(self._cumulative, index)
        previous_total = self._cumulative[source_index - 1] if source_index > 0 else 0
        return source_index, index - previous_total

    @overload
    def __getitem__(self, index: int) -> Atoms: ...

    @overload
    def __getitem__(self, index: slice) -> Trajectory: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> Atoms | Trajectory:
        if isinstance(index, slice):
            paths: list[Path] = []
            offsets: list[list[int]] = []

            for frame_index in range(*index.indices(len(self))):
                source_index, local_index = self._locate(frame_index)
                path = self._paths[source_index]
                offset = self._offsets[source_index][local_index]

                if paths and paths[-1] == path:
                    offsets[-1].append(offset)
                else:
                    paths.append(path)
                    offsets.append([offset])

            return Trajectory._from_parts(
                paths=paths,
                offsets=offsets,
                reader=self._reader,
            )

        if index < 0:
            index += len(self)

        if not 0 <= index < len(self):
            raise IndexError("trajectory frame index out of range")

        source_index, local_index = self._locate(index)
        return self._reader(
            self._paths[source_index],
            self._offsets[source_index][local_index],
        )

    def __iter__(self) -> Iterator[Atoms]:
        """Iterate lazily over trajectory frames."""
        for path, offsets in zip(self._paths, self._offsets, strict=True):
            for offset in offsets:
                yield self._reader(path, offset)

    def __add__(self, other: object) -> Trajectory:
        """Return a lazy concatenation with another trajectory."""
        if not isinstance(other, Trajectory):
            return NotImplemented

        if not _same_reader(self._reader, other._reader):
            raise ValueError("can only concatenate trajectories with compatible readers")

        return Trajectory._from_parts(
            paths=self._paths + other._paths,
            offsets=self._offsets + other._offsets,
            reader=self._reader,
        )

    def __radd__(self, other: object) -> Trajectory:
        """Support ``sum([...])`` by treating integer zero as the identity."""
        if isinstance(other, int) and other == 0:
            return self

        if isinstance(other, Trajectory):
            return other + self

        return NotImplemented

    def __repr__(self) -> str:
        if len(self._paths) == 1:
            return f"{type(self).__name__}(path={str(self.path)!r}, n_frames={len(self)})"
        return (
            f"{type(self).__name__}("
            f"n_files={len(self._paths)}, n_frames={len(self)})"
        )

    def __str__(self) -> str:
        return self.__repr__()
