"""The paper's Eq. 20-22 correction for an hourly surface heat transfer coefficient.

Without it, the sol-air temperature divides absorbed solar by the HOURLY h_e while the solver
couples the surface to the environment through a FIXED 1/R_so. The two disagree by up to a
factor of 2.3 on a calm sunny afternoon, which lands squarely on the daily peaks that comfort
metrics read.

The reference here is the finite-difference solver driven by the PHYSICAL boundary condition
h_e(t)*(T_env - T_s) + alpha*Q, so it never assumes a constant resistance at all — it is an
independent arbiter rather than a re-derivation of the correction being tested.
"""

from __future__ import annotations

import numpy as np
import pytest

from surface_temps.admittance import (
    solve_surface_temperature,
    solve_surface_temperature_variable_h,
)
from surface_temps.materials import Assembly, Layer

from .reference_fd import FdLayer, solve_fd

SLAB = dict(thickness=0.20, conductivity=1.40, density=2100, specific_heat=840)


def _scenario(n=24 * 10, calm_afternoons=True):
    """A week of warm weather with strong sun and a wind pattern that goes calm by day."""
    t = np.arange(n, dtype=float)
    hour = t % 24.0
    T_env = 22.0 + 7.0 * np.sin(2.0 * np.pi * (hour - 9.0) / 24.0)
    Q_absorbed = 0.7 * np.maximum(0.0, 850.0 * np.sin(np.pi * (hour - 6.0) / 12.0))
    # Wind: brisk at night, calm through the afternoon — the case that breaks a fixed R_so.
    wind = np.where((hour > 10) & (hour < 18), 0.3, 4.5) if calm_afternoons else np.full(n, 2.0)
    h_c = 5.7 + 3.8 * wind
    h_e = h_c + 5.0
    return T_env, Q_absorbed, h_e


def _fd_truth(T_env, Q_absorbed, h_e, assembly, T_internal, n_cells=40, substeps=20):
    layers = [
        FdLayer(l.thickness, l.conductivity, l.density, l.specific_heat, n_cells)
        for l in reversed(assembly.layers)
    ]
    return solve_fd(
        T_env, layers, R_so=1.0 / float(np.mean(h_e)), R_si=assembly.R_si,
        T_internal=T_internal, substeps=substeps, n_cycles=8,
        h_e_series=h_e, q_absorbed_series=Q_absorbed,
    )


def test_correction_beats_a_fixed_resistance_against_the_physical_reference():
    T_env, Q_absorbed, h_e = _scenario()
    assembly = Assembly(layers=[Layer(**SLAB)], R_si=0.13, R_so=0.04)
    T_internal = 22.0

    truth = _fd_truth(T_env, Q_absorbed, h_e, assembly, T_internal)

    # The old path: sol-air divided by hourly h_e, solved against the fixed R_so.
    T_sol_old = T_env + Q_absorbed / h_e
    old = solve_surface_temperature(T_sol_old, assembly, T_internal)
    new = solve_surface_temperature_variable_h(T_env, Q_absorbed, h_e, assembly, T_internal)

    rmse_old = float(np.sqrt(np.mean((old - truth) ** 2)))
    rmse_new = float(np.sqrt(np.mean((new - truth) ** 2)))
    peak_old = float(np.max(np.abs(old - truth)))
    peak_new = float(np.max(np.abs(new - truth)))

    assert rmse_new < rmse_old, f"correction made it worse: {rmse_new:.3f} vs {rmse_old:.3f}"
    assert peak_new < peak_old, f"peak error not improved: {peak_new:.2f} vs {peak_old:.2f}"


def test_correction_is_a_no_op_when_h_is_already_constant():
    """With a constant h_e the correction must reduce exactly to the plain solve."""
    n = 24 * 6
    t = np.arange(n, dtype=float)
    T_env = 20.0 + 6.0 * np.sin(2.0 * np.pi * t / 24.0)
    Q_absorbed = np.maximum(0.0, 500.0 * np.sin(2.0 * np.pi * (t % 24) / 24.0 - 1.0))
    h_e = np.full(n, 25.0)
    assembly = Assembly(layers=[Layer(**SLAB)], R_si=0.13, R_so=1.0 / 25.0)

    plain = solve_surface_temperature(T_env + Q_absorbed / 25.0, assembly, 20.0)
    corrected = solve_surface_temperature_variable_h(T_env, Q_absorbed, h_e, assembly, 20.0)

    assert np.max(np.abs(plain - corrected)) < 1e-9


def test_insulated_roof_stays_bounded():
    """The divergence guard: an insulated build-up ran away to 3e4 degC before damping.

    The fixed point has gain |dh|/h_bar, which exceeds 1 for a realistic wind series, so any
    surface that closely tracks its driving temperature diverges without under-relaxation.
    """
    from surface_temps.materials import concrete_roof

    T_env, Q_absorbed, h_e = _scenario()
    result = solve_surface_temperature_variable_h(
        T_env, Q_absorbed, h_e, concrete_roof(), T_internal=22.0
    )

    assert np.all(np.isfinite(result))
    assert result.max() < 120.0 and result.min() > -50.0, (
        f"unbounded: {result.min():.1f}..{result.max():.1f}"
    )


def test_rejects_non_positive_h():
    T_env, Q_absorbed, h_e = _scenario(n=48)
    h_e = h_e.copy()
    h_e[3] = 0.0
    with pytest.raises(ValueError, match="positive"):
        solve_surface_temperature_variable_h(
            T_env, Q_absorbed, h_e, Assembly(layers=[Layer(**SLAB)], R_si=0.13), 20.0
        )


def test_calm_sunny_peaks_are_where_it_matters():
    """Document the direction: a fixed R_so over-injects solar when the air is still."""
    T_env, Q_absorbed, h_e = _scenario()
    assembly = Assembly(layers=[Layer(**SLAB)], R_si=0.13, R_so=0.04)

    old = solve_surface_temperature(T_env + Q_absorbed / h_e, assembly, 22.0)
    new = solve_surface_temperature_variable_h(T_env, Q_absorbed, h_e, assembly, 22.0)

    # Calm hours are the afternoon window in the scenario.
    hour = np.arange(len(old)) % 24
    calm = (hour > 10) & (hour < 18)
    assert old[calm].max() > new[calm].max(), (
        "with a fixed R_so the calm sunny peak should be the OVER-predicted one"
    )
