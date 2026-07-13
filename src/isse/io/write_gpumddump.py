from __future__ import annotations

from pathlib import Path

from ..structures import Atoms, Trajectory


def write_gpumd_dump(
    filename: str | Path,
    trajectory: Atoms | Trajectory | list[Atoms],
) -> None:
    """
    Write a GPUMD/extended-XYZ style dump trajectory.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path of the output file.
    trajectory : Atoms, Trajectory, or list of Atoms
        Atomic configuration or trajectory whose frames are written in
        GPUMD/extended-XYZ format. A list of ``Atoms`` is accepted for
        in-memory trajectories. Each frame must provide symbols, cell and
        Cartesian positions. Optional velocities, forces, unwrapped positions,
        masses, groups and timestep metadata are written when present.

    Returns
    -------
    None
        The function writes the file and returns ``None``.
    """
    with Path(filename).open("w") as file:
        for iframe, atoms in enumerate(_iter_trajectory(trajectory)):
            properties = ["species:S:1", "pos:R:3"]
            if atoms.velocities is not None:
                properties.append("vel:R:3")
            if atoms.forces is not None:
                properties.append("force:R:3")
            if atoms.unwrapped_positions is not None:
                properties.append("unwrapped_position:R:3")
            if atoms.masses is not None:
                properties.append("mass:R:1")
            if "groups" in atoms.arrays:
                properties.append("group:I:1")

            timestep = atoms.info.get("timestep", str(iframe))
            lattice = " ".join(f"{value:.16g}" for value in atoms.cell.reshape(-1))
            file.write(f"{len(atoms)}\n")
            file.write(
                f'Lattice="{lattice}" Properties={":".join(properties)} '
                f"Time={timestep}\n"
            )

            for iatom, symbol in enumerate(atoms.symbols):
                values = [symbol]
                values.extend(f"{value:.16g}" for value in atoms.positions[iatom])
                if atoms.velocities is not None:
                    values.extend(f"{value:.16g}" for value in atoms.velocities[iatom])
                if atoms.forces is not None:
                    values.extend(f"{value:.16g}" for value in atoms.forces[iatom])
                if atoms.unwrapped_positions is not None:
                    values.extend(
                        f"{value:.16g}" for value in atoms.unwrapped_positions[iatom]
                    )
                if atoms.masses is not None:
                    values.append(f"{atoms.masses[iatom]:.16g}")
                if "groups" in atoms.arrays:
                    values.append(str(int(atoms.arrays["groups"][iatom])))
                file.write(" ".join(values) + "\n")


def _iter_trajectory(
    trajectory: Atoms | Trajectory | list[Atoms],
) -> tuple[Atoms, ...] | list[Atoms] | Trajectory:
    if isinstance(trajectory, Atoms):
        return (trajectory,)
    return trajectory
