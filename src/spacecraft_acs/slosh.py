"""Propellant slosh as equivalent mechanical modes.

Each tank's first lateral slosh mode is modeled as a spring-mass-damper
(the standard equivalent-mechanical model, Abramson NASA SP-106 / Dodge).
For torque-free flight, eliminating the spacecraft translational DOF by
momentum conservation reduces a slosh mass m_s at tank position r moving
laterally along ê to exactly the hybrid-coordinate modal form used for the
structural modes:

    effective (mass-normalized) participation  δ = (r × ê) · sqrt(m_s / (1 − m_s/M))
    effective modal frequency                   ω̃ = ω_tank / sqrt(1 − m_s/M)

where M is the total spacecraft mass and ω_tank is the slosh frequency in
tank-fixed axes. Two orthogonal lateral directions per tank give two modes.
The sqrt(1/(1 − m_s/M)) factors carry the CM-shift coupling; they matter for
slosh masses that are a non-trivial fraction of the spacecraft.

The participating slosh mass fraction comes from a fit to the SP-106
spherical-tank first-mode data as a function of fill fraction (nearly all
of a shallow pool sloshes; almost none of a full tank does):

    m_s / m_prop ≈ (1 − f) (1 + 0.15 f),   f = fill fraction

Frequency and damping are direct inputs: in the GEO regime they are set by
surface tension / PMD geometry (zero-g) or thrust acceleration (burns), both
outside this model's scope to predict — and both poorly predictable in
general, which is why the Monte Carlo disperses them aggressively.
"""

from __future__ import annotations

import numpy as np


def slosh_mass_fraction(fill_fraction: float) -> float:
    """First-lateral-mode participating mass fraction for a spherical tank.
    Approximate fit to NASA SP-106 data."""
    if not 0.0 < fill_fraction < 1.0:
        raise ValueError("fill_fraction must be in (0, 1)")
    return (1.0 - fill_fraction) * (1.0 + 0.15 * fill_fraction)


def tank_equivalent_modes(tank, total_mass: float) -> list:
    """Reduce one tank to two equivalent rotational modes (the two lateral
    slosh directions). Returns ModeConfig instances."""
    from .config import ModeConfig  # deferred to avoid a circular import

    m_s = tank.slosh_mass
    if m_s is None:
        m_s = tank.propellant_mass * slosh_mass_fraction(tank.fill_fraction)
    if m_s >= total_mass:
        raise ValueError("slosh mass must be less than the spacecraft mass")
    cm_factor = 1.0 - m_s / total_mass

    axis = np.asarray(tank.axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    # Two lateral unit vectors orthogonal to the tank axis
    seed = np.array([1.0, 0.0, 0.0])
    if abs(axis @ seed) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    r = np.asarray(tank.location, dtype=float)
    scale = np.sqrt(m_s / cm_factor)
    freq = tank.freq_hz / np.sqrt(cm_factor)
    return [
        ModeConfig(
            freq_hz=freq,
            damping=tank.damping,
            participation=np.cross(r, e) * scale,
        )
        for e in (e1, e2)
    ]
