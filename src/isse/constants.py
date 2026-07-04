from __future__ import annotations

from numpy import asarray
from numpy.typing import NDArray

# NIST CODATA: https://physics.nist.gov/cuu/Constants/index.html
ANGSTROM_TO_BOHR = 1.8897259886
BOHR_TO_ANGSTROM = 0.5291772106
AMU_A2_FS2_TO_EV = 103.64269653
KB_EV_K = 8.617333262e-5

# Approximate standard atomic weights in unified atomic mass units (u).
# Values are intended for generic numerical use, not for isotope-resolved
# calculations or metrological reference work.
ATOMIC_MASSES = {
    "H": 1.008,
    "He": 4.002602,
    "Li": 6.94,
    "Be": 9.0121831,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403163,
    "Ne": 20.1797,
    "Na": 22.98976928,
    "Mg": 24.305,
    "Al": 26.9815385,
    "Si": 28.085,
    "P": 30.973761998,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K": 39.0983,
    "Ca": 40.078,
    "Sc": 44.955908,
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.938044,
    "Fe": 55.845,
    "Co": 58.933194,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.63,
    "As": 74.921595,
    "Se": 78.971,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.4678,
    "Sr": 87.62,
    "Y": 88.90584,
    "Zr": 91.224,
    "Nb": 92.90637,
    "Mo": 95.95,
    "Tc": 98.0,
    "Ru": 101.07,
    "Rh": 102.9055,
    "Pd": 106.42,
    "Ag": 107.8682,
    "Cd": 112.414,
    "In": 114.818,
    "Sn": 118.71,
    "Sb": 121.76,
    "Te": 127.6,
    "I": 126.90447,
    "Xe": 131.293,
    "Cs": 132.90545196,
    "Ba": 137.327,
    "La": 138.90547,
    "Ce": 140.116,
    "Pr": 140.90766,
    "Nd": 144.242,
    "Pm": 145.0,
    "Sm": 150.36,
    "Eu": 151.964,
    "Gd": 157.25,
    "Tb": 158.92535,
    "Dy": 162.5,
    "Ho": 164.93033,
    "Er": 167.259,
    "Tm": 168.93422,
    "Yb": 173.045,
    "Lu": 174.9668,
    "Hf": 178.49,
    "Ta": 180.94788,
    "W": 183.84,
    "Re": 186.207,
    "Os": 190.23,
    "Ir": 192.217,
    "Pt": 195.084,
    "Au": 196.966569,
    "Hg": 200.592,
    "Tl": 204.38,
    "Pb": 207.2,
    "Bi": 208.9804,
    "Po": 209.0,
    "At": 210.0,
    "Rn": 222.0,
    "Fr": 223.0,
    "Ra": 226.0,
    "Ac": 227.0,
    "Th": 232.0377,
    "Pa": 231.03588,
    "U": 238.02891,
    "Np": 237.0,
    "Pu": 244.0,
    "Am": 243.0,
    "Cm": 247.0,
    "Bk": 247.0,
    "Cf": 251.0,
    "Es": 252.0,
    "Fm": 257.0,
    "Md": 258.0,
    "No": 259.0,
    "Lr": 262.0,
    "Rf": 267.0,
    "Db": 270.0,
    "Sg": 271.0,
    "Bh": 270.0,
    "Hs": 277.0,
    "Mt": 278.0,
    "Ds": 281.0,
    "Rg": 282.0,
    "Cn": 285.0,
    "Nh": 286.0,
    "Fl": 289.0,
    "Mc": 290.0,
    "Lv": 293.0,
    "Ts": 294.0,
    "Og": 294.0,
}


def mass_from_symbol(symbol: str) -> float:
    """
    Return the approximate atomic weight corresponding to a chemical symbol.

    Parameters
    ----------
    symbol : str
        Chemical symbol of the element.

    Returns
    -------
    float
        Approximate atomic weight in unified atomic mass units (u).

    Raises
    ------
    KeyError
        If `symbol` is not present in the atomic weight table.
    """

    try:
        return ATOMIC_MASSES[symbol]
    except KeyError as exc:
        raise KeyError(f"Unknown chemical symbol: {symbol!r}") from exc


def masses_from_symbols(symbols: list[str]) -> NDArray:
    """
    Return approximate atomic weights corresponding to chemical symbols.

    Parameters
    ----------
    symbols : list[str]
        Chemical symbols of the elements.

    Returns
    -------
    NDArray
        Approximate atomic weights in unified atomic mass units (u).
    """

    return asarray([mass_from_symbol(symbol) for symbol in symbols], dtype=float)


def symbol_from_mass(mass: float, tolerance: float = 1.0e-3) -> str:
    """
    Return the chemical symbol corresponding to an approximate atomic weight.

    Parameters
    ----------
    mass : float
        Approximate atomic weight in unified atomic mass units (u).
    tolerance : float, optional
        Absolute tolerance used when comparing `mass` with tabulated atomic
        weights.

    Returns
    -------
    str
        Chemical symbol of the element.

    Raises
    ------
    ValueError
        If no chemical symbol corresponds to the given mass within `tolerance`,
        or if multiple symbols match the given mass.
    """

    mass = float(mass)
    tolerance = float(tolerance)

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    matches = [
        (symbol, tabulated_mass)
        for symbol, tabulated_mass in ATOMIC_MASSES.items()
        if abs(tabulated_mass - mass) <= tolerance
    ]

    if len(matches) == 1:
        return matches[0][0]

    if len(matches) > 1:
        formatted = ", ".join(
            f"{symbol}={tabulated_mass:g}" for symbol, tabulated_mass in matches
        )
        raise ValueError(
            f"Mass {mass:g} matches multiple elements within tolerance "
            f"{tolerance:g}: {formatted}"
        )

    nearest_symbol, nearest_mass = min(
        ATOMIC_MASSES.items(),
        key=lambda item: abs(item[1] - mass),
    )
    raise ValueError(
        f"No chemical symbol found for mass {mass:g} within tolerance "
        f"{tolerance:g}. Nearest is {nearest_symbol}={nearest_mass:g}."
    )


def symbols_from_masses(masses: NDArray, tolerance: float = 1.0e-3) -> list[str]:
    """
    Return chemical symbols corresponding to approximate atomic weights.

    Parameters
    ----------
    masses : NDArray
        Approximate atomic weights in unified atomic mass units (u).
    tolerance : float, optional
        Absolute tolerance used when comparing masses with tabulated atomic
        weights.

    Returns
    -------
    list[str]
        Chemical symbols corresponding to the atomic weights.

    Raises
    ------
    ValueError
        If any mass matches no chemical symbol, or if any mass matches multiple
        symbols within `tolerance`.
    """

    return [symbol_from_mass(mass, tolerance=tolerance) for mass in masses]
