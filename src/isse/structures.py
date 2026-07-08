from __future__ import annotations

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


class Trajectory(Sequence[Atoms]):
    """
    Provide lazy access to frames in an atomistic trajectory.

    Frames are read from disk only when accessed. The trajectory stores the
    byte offset of each frame rather than one :class:`Atoms` instance per
    timestep.

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

    Slices return lazy :class:`TrajectoryView` objects and therefore do not
    immediately load the selected frames.

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

    __slots__ = ("_path", "_offsets", "_reader")

    def __init__(
        self,
        path: str | Path,
        offsets: Sequence[int],
        reader: FrameReader,
    ) -> None:

        self._path = Path(path)
        self._offsets = tuple(offsets)
        self._reader = reader

        if not self._path.is_file():
            raise FileNotFoundError(self._path)

        if any(offset < 0 for offset in self._offsets):
            raise ValueError("frame offsets must be non-negative")

    @property
    def path(self) -> Path:
        """Return the path to the trajectory file."""
        return self._path

    def __len__(self) -> int:
        """Return the number of frames."""
        return len(self._offsets)

    @overload
    def __getitem__(self, index: int) -> Atoms: ...

    @overload
    def __getitem__(self, index: slice) -> Trajectory: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> Atoms | Trajectory:
        if isinstance(index, slice):
            return Trajectory(
                path=self._path,
                offsets=self._offsets[index],
                reader=self._reader,
            )

        if index < 0:
            index += len(self)

        if not 0 <= index < len(self):
            raise IndexError("trajectory frame index out of range")

        return self._reader(
            self._path,
            self._offsets[index],
        )

    def __iter__(self) -> Iterator[Atoms]:
        """Iterate lazily over trajectory frames."""
        for offset in self._offsets:
            yield self._reader(self._path, offset)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={str(self.path)!r}, n_frames={len(self)})"

    def __str__(self) -> str:
        return self.__repr__()
