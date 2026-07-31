"""Regression tests for the driving-signal defects fixed in the paper audit.

Each of these exists because ``scripts/mutation_check.py`` proved the suite could NOT tell the
fixed code from the broken code: reintroducing the emissivity divisor, hardcoding h_r, or
silently disabling the Eq. 20-22 correction all left the suite green. A fix nobody can regress
into is worth more than the fix itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from surface_temps.admittance import (
    solve_surface_temperature,
    solve_surface_temperature_variable_h,
)
from surface_temps.constants import STEFAN_BOLTZMANN
from surface_temps.materials import Assembly, Layer
from surface_temps.solar import h_radiative, sky_temperature
from surface_temps.weather import load_epw

from .reference_fd import FdLayer, solve_fd

EPW = "data/atlanta_tmy3.epw"
SLAB = dict(thickness=0.20, conductivity=1.40, density=2100, specific_heat=840)


@pytest.fixture(scope="module")
def weather():
    return load_epw(EPW)


# --- sky temperature -------------------------------------------------------------------


def test_sky_temperature_is_the_blackbody_equivalent():
    """T_sky = (IR/sigma)^0.25, with NO emissivity divisor.

    The EPW horizontal-IR field IS the downwelling long-wave flux, so the sky's emissivity is
    already inside it. Dividing by 0.90 inflates T_sky by (1/0.9)^0.25 = 1.0267 in kelvin.
    """

    class _Fake:
        infrared_horizontal = np.array([300.0, 400.0])

    result = sky_temperature(_Fake())
    expected = (np.array([300.0, 400.0]) / STEFAN_BOLTZMANN) ** 0.25 - 273.15
    assert np.allclose(result, expected, atol=1e-9)

    # And confirm the wrong form is meaningfully different, so this cannot pass vacuously.
    inflated = (np.array([300.0]) / (0.90 * STEFAN_BOLTZMANN)) ** 0.25 - 273.15
    assert inflated[0] - result[0] > 6.0


def test_sky_sits_below_air_all_year(weather):
    """The physical signature the divisor destroyed.

    A real sky is colder than the air — that depression is what drives night-time radiative
    cooling. With the 0.90 divisor the modelled sky was WARMER than the air for about a third
    of the year, which an IR-derived sky temperature cannot be.
    """
    T_sky = sky_temperature(weather)
    T_air = np.asarray(weather.temp_air, dtype=float)

    warmer_hours = int(np.sum(T_sky > T_air))
    assert warmer_hours == 0, f"sky warmer than air for {warmer_hours} hours — divisor is back"

    depression = float(np.mean(T_air - T_sky))
    assert 7.0 < depression < 14.0, f"mean clear-sky depression {depression:.1f} K is implausible"


# --- radiative coefficient -------------------------------------------------------------


@pytest.mark.parametrize(
    "emissivity,expected",
    [(0.95, 5.25), (0.90, 4.98), (0.25, 1.38), (0.05, 0.28)],
)
def test_h_radiative_scales_with_emissivity(emissivity, expected):
    """h_r = 4*eps*sigma*T^3 at 290 K. A hardcoded 5.0 is right ONLY near eps = 0.90."""
    assert h_radiative(emissivity, 16.85) == pytest.approx(expected, abs=0.02)


def test_h_radiative_actually_depends_on_emissivity():
    """Kills the 'return a constant 5.0' regression directly."""
    low = h_radiative(0.05, 16.85)
    high = h_radiative(0.95, 16.85)
    assert high / low == pytest.approx(0.95 / 0.05, rel=1e-9)
    assert low < 1.0, "a bare-metal surface must have a small radiative coefficient"


def test_low_emissivity_surface_runs_hotter_than_a_hardcoded_hr_would_predict():
    """The consequence, end to end: less radiative loss means a hotter metal surface."""
    n = 24 * 7
    t = np.arange(n, dtype=float)
    T_air = 28.0 + 6.0 * np.sin(2.0 * np.pi * t / 24.0 - 1.2)
    solar = np.maximum(0.0, 850.0 * np.sin(2.0 * np.pi * (t % 24) / 24.0 - 1.2))
    h_c = np.full(n, 12.0)
    assembly = Assembly(layers=[Layer(**SLAB)], R_si=0.13, R_so=0.04)

    def peak(h_r):
        h_e = h_c + h_r
        T_sol = (h_c * T_air + h_r * (T_air - 9.0) + 0.55 * solar) / h_e
        return solve_surface_temperature(T_sol, assembly, 22.0).max()

    correct = peak(h_radiative(0.25, T_air))  # pre-weathered zinc
    hardcoded = peak(np.full(n, 5.0))
    assert correct - hardcoded > 5.0, (
        "a low-emissivity surface must come out markedly hotter once h_r follows its emissivity"
    )


# --- variable convection ---------------------------------------------------------------


def test_qco_correction_carries_the_improvement():
    """The Eq. 20-22 correction must beat the SAME solve with q_co removed.

    Comparing against the old fixed-R_so path is not enough: switching R_so to 1/mean(h_e)
    changes the answer on its own, so a mutation that zeroes q_co still looked fine. This
    isolates the corrective flux itself.
    """
    n = 24 * 10
    t = np.arange(n, dtype=float)
    hour = t % 24.0
    T_env = 22.0 + 7.0 * np.sin(2.0 * np.pi * (hour - 9.0) / 24.0)
    Q_absorbed = 0.7 * np.maximum(0.0, 850.0 * np.sin(np.pi * (hour - 6.0) / 12.0))
    wind = np.where((hour > 10) & (hour < 18), 0.3, 4.5)
    h_e = (5.7 + 3.8 * wind) + 5.0

    assembly = Assembly(layers=[Layer(**SLAB)], R_si=0.13, R_so=0.04)
    h_bar = float(np.mean(h_e))

    truth = solve_fd(
        T_env,
        [FdLayer(l.thickness, l.conductivity, l.density, l.specific_heat, 40)
         for l in reversed(assembly.layers)],
        R_so=1.0 / h_bar, R_si=assembly.R_si, T_internal=22.0,
        substeps=20, n_cycles=8, h_e_series=h_e, q_absorbed_series=Q_absorbed,
    )

    corrected = solve_surface_temperature_variable_h(T_env, Q_absorbed, h_e, assembly, 22.0)

    # Exactly what the solver reduces to when q_co is dropped: mean-h resistance, no correction.
    from dataclasses import replace

    uncorrected = solve_surface_temperature(
        T_env + Q_absorbed / h_bar, replace(assembly, R_so=1.0 / h_bar), 22.0
    )

    err_corrected = float(np.max(np.abs(corrected - truth)))
    err_uncorrected = float(np.max(np.abs(uncorrected - truth)))

    assert err_corrected < err_uncorrected, (
        f"the q_co correction is not doing anything: {err_corrected:.3f} K vs "
        f"{err_uncorrected:.3f} K against the physical reference"
    )
    # It should be a substantial improvement, not noise.
    assert err_uncorrected - err_corrected > 0.5


def test_correction_changes_the_answer_at_all():
    """A blunt guard: with genuinely variable h, q_co must move the result."""
    n = 24 * 6
    t = np.arange(n, dtype=float)
    T_env = 20.0 + 8.0 * np.sin(2.0 * np.pi * t / 24.0)
    Q_absorbed = np.maximum(0.0, 600.0 * np.sin(2.0 * np.pi * (t % 24) / 24.0 - 1.0))
    h_e = np.where((t % 24) > 12, 40.0, 11.0)
    assembly = Assembly(layers=[Layer(**SLAB)], R_si=0.13, R_so=0.04)

    from dataclasses import replace

    h_bar = float(np.mean(h_e))
    corrected = solve_surface_temperature_variable_h(T_env, Q_absorbed, h_e, assembly, 20.0)
    uncorrected = solve_surface_temperature(
        T_env + Q_absorbed / h_bar, replace(assembly, R_so=1.0 / h_bar), 20.0
    )

    assert np.max(np.abs(corrected - uncorrected)) > 1.0, "q_co had no effect"
