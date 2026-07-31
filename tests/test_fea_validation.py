"""Validate the frequency-domain solver against an independent time-domain solver.

This is the check the paper performs in its Figure 3, and it is the strongest evidence in the
suite: ``tests/reference_fd.py`` integrates the heat equation on a node network in the TIME
domain and shares no code with ``surface_temps``. If the admittance transfer function, the
layer matrices, the layer ordering, the R_so boundary condition or the FFT normalisation were
wrong, these tests would fail.

One subtlety governs the tolerances. The two methods interpolate BETWEEN hourly samples
differently — the finite-difference solver linearly, the frequency method trigonometrically —
so for a signal that is not band-limited they are genuinely driven by different sub-hour inputs
and disagree by a small amount that does NOT vanish under grid refinement. Tests that need a
tight tolerance therefore hand the reference solver the exact analytic driver via ``T_env_fn``;
the sawtooth test, which reproduces the paper's own excitation, keeps a looser tolerance and
documents why.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from surface_temps.admittance import solve_surface_temperature
from surface_temps.materials import Assembly, Layer, brick_wall, concrete_roof

from .reference_fd import FdLayer, sawtooth, solve_fd

# Paper Table 1: an idealised 200 mm cast concrete slab.
CAST_CONCRETE = dict(thickness=0.20, conductivity=1.40, density=2100, specific_heat=840)
R_SO = 0.04
R_SI = 0.13


def _fd_for(assembly: Assembly, T_driving, n_cells=80, substeps=80, T_env_fn=None):
    """Run the reference solver for an Assembly.

    ``Assembly.layers`` runs INSIDE to OUTSIDE; the finite-difference network runs OUTER to
    INNER, so the list is reversed. Getting this backwards is not a subtle error — it moved
    the surface response by 0.48 K when this harness was first written.
    """
    layers = [
        FdLayer(layer.thickness, layer.conductivity, layer.density, layer.specific_heat, n_cells)
        for layer in reversed(assembly.layers)
    ]
    return solve_fd(
        np.asarray(T_driving, dtype=float),
        layers,
        R_so=assembly.R_so,
        R_si=assembly.R_si,
        T_internal=0.0,
        substeps=substeps,
        T_env_fn=T_env_fn,
    )


def test_sinusoid_matches_finite_difference_to_micro_kelvin():
    """A pure 24 h sinusoid on a concrete slab: the two methods must agree essentially exactly."""
    n = 24 * 8
    assembly = Assembly(layers=[Layer(**CAST_CONCRETE)], R_si=R_SI, R_so=R_SO)
    driver = lambda t: np.sin(2.0 * np.pi * t / 24.0)  # noqa: E731
    T_driving = driver(np.arange(n, dtype=float))

    T_freq = solve_surface_temperature(T_driving, assembly, T_internal=0.0)
    T_fd = _fd_for(assembly, T_driving, n_cells=80, substeps=80, T_env_fn=driver)

    assert np.max(np.abs(T_freq - T_fd)) < 1e-4


def test_finite_difference_agreement_converges_second_order():
    """Refining the reference grid must shrink the gap ~4x, proving it is discretisation error.

    A constant residual under refinement would mean the two solvers disagree on the PHYSICS;
    a fourfold drop per refinement means they agree and the reference is simply discrete.
    """
    n = 24 * 8
    assembly = Assembly(layers=[Layer(**CAST_CONCRETE)], R_si=R_SI, R_so=R_SO)
    driver = lambda t: np.sin(2.0 * np.pi * t / 24.0)  # noqa: E731
    T_driving = driver(np.arange(n, dtype=float))
    T_freq = solve_surface_temperature(T_driving, assembly, T_internal=0.0)

    errors = []
    for cells in (20, 40, 80):
        T_fd = _fd_for(assembly, T_driving, n_cells=cells, substeps=cells, T_env_fn=driver)
        errors.append(float(np.max(np.abs(T_freq - T_fd))))

    assert errors[0] > errors[1] > errors[2], f"not converging: {errors}"
    for coarse, fine in itertools.pairwise(errors):
        assert 3.0 < coarse / fine < 5.0, f"expected ~4x per refinement, got {coarse / fine:.2f}"


@pytest.mark.parametrize(
    "factory", [brick_wall, concrete_roof], ids=["brick_wall", "concrete_roof"]
)
def test_real_multilayer_assemblies_match_finite_difference(factory):
    """Multi-layer build-ups agree too — this is what pins the layer ORDER and the interfaces."""
    n = 24 * 8
    assembly = factory()
    driver = lambda t: 10.0 * np.sin(2.0 * np.pi * t / 24.0)  # noqa: E731
    T_driving = driver(np.arange(n, dtype=float))

    T_freq = solve_surface_temperature(T_driving, assembly, T_internal=0.0)
    T_fd = _fd_for(assembly, T_driving, n_cells=80, substeps=120, T_env_fn=driver)

    assert np.max(np.abs(T_freq - T_fd)) < 5e-3


def test_reversing_layer_order_is_detectable():
    """Guard the guard: if the harness ignored layer order, the test above could not fail."""
    n = 24 * 8
    assembly = concrete_roof()  # strongly asymmetric: insulation near the outside
    driver = lambda t: 10.0 * np.sin(2.0 * np.pi * t / 24.0)  # noqa: E731
    T_driving = driver(np.arange(n, dtype=float))

    correct = _fd_for(assembly, T_driving, n_cells=40, substeps=60, T_env_fn=driver)
    flipped_layers = [
        FdLayer(layer.thickness, layer.conductivity, layer.density, layer.specific_heat, 40)
        for layer in assembly.layers  # deliberately NOT reversed
    ]
    flipped = solve_fd(
        T_driving,
        flipped_layers,
        R_so=assembly.R_so,
        R_si=assembly.R_si,
        T_internal=0.0,
        substeps=60,
        T_env_fn=driver,
    )

    assert np.max(np.abs(correct - flipped)) > 0.1, "layer order has no effect — harness is inert"


def test_paper_figure3_sawtooth():
    """Reproduce the paper's Figure 3 comparison: sawtooth driver, concrete slab.

    Tolerance is looser than the sinusoid tests on purpose. A sawtooth is not band-limited, so
    the trigonometric reconstruction the frequency method applies between hourly samples differs
    from the reference solver's linear interpolation; the residual is that difference, not a
    disagreement about the physics (it does not shrink under grid refinement).
    """
    n = 24 * 8
    assembly = Assembly(layers=[Layer(**CAST_CONCRETE)], R_si=R_SI, R_so=R_SO)
    T_driving = sawtooth(n, amplitude=1.0, period_steps=24)

    T_freq = solve_surface_temperature(T_driving, assembly, T_internal=0.0)
    T_fd = _fd_for(assembly, T_driving, n_cells=60, substeps=60)

    assert np.max(np.abs(T_freq - T_fd)) < 0.02
    # The physical content of the paper's figure: a 200 mm concrete slab damps the swing.
    amp_in = (T_driving.max() - T_driving.min()) / 2
    amp_out = (T_freq.max() - T_freq.min()) / 2
    assert 0.5 < amp_out / amp_in < 0.8
