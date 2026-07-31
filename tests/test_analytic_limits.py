"""Cases where the answer is known in closed form.

These are cheap, exact, and they fail loudly if a sign, a reciprocal or a layer order is wrong.
Several of them would pass even for a badly broken solver if written carelessly, so each one
states the physical value it pins rather than merely asserting self-consistency.
"""

from __future__ import annotations

import numpy as np
import pytest

from surface_temps.admittance import solve_surface_temperature
from surface_temps.materials import Assembly, Layer

R_SI = 0.13
R_SO = 0.04

CONCRETE = dict(conductivity=1.40, density=2100, specific_heat=840)


def _H(assembly: Assembly, period_s: float) -> complex:
    """The solver's own surface transfer function at one period."""
    M = assembly.total_matrix(period_s)
    return 1.0 / (1.0 + M[0, 0] * assembly.R_so / M[0, 1])


def test_layer_matrix_determinant_is_unity():
    """det(M) = 1 for every layer at every period — the defining property of the 2x2 form.

    This is what catches a mis-typed sinh/cosh or a lost factor in z2/z3: the paper's Eq. 4
    written with xi(1+j) instead of the physical lambda*p still satisfies det = 1, so this
    test alone is not sufficient, but its failure is unambiguous.
    """
    layer = Layer(thickness=0.15, **CONCRETE)
    for period in (3600.0, 86400.0, 86400.0 * 30, 86400.0 * 365):
        M = layer.transfer_matrix(period)
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        assert det == pytest.approx(1.0, abs=1e-9), f"det={det} at P={period}"


def test_layer_matrix_uses_physical_admittance_not_the_papers_xi():
    """z3 must be lambda*p*sinh(pL), i.e. the ISO 13786 admittance.

    The paper's Eq. 4/5 as printed give z3 = xi(1+j)sinh with xi = sqrt(2*pi*lambda*rho*c/P),
    which is sqrt(2) LARGER than lambda*p. Both forms preserve det(M) = 1, so only a direct
    check of the element itself distinguishes them.
    """
    layer = Layer(thickness=0.15, **CONCRETE)
    period = 86400.0
    M = layer.transfer_matrix(period)

    lam, rho, c = layer.conductivity, layer.density, layer.specific_heat
    omega = 2.0 * np.pi / period
    p = np.sqrt(1j * omega / (lam / (rho * c)))
    expected_z3 = lam * p * np.sinh(p * layer.thickness)
    xi_form = np.sqrt(2 * np.pi * lam * rho * c / period) * (1 + 1j) * np.sinh(p * layer.thickness)

    assert M[1, 0] == pytest.approx(expected_z3, rel=1e-12)
    # And confirm the two really do differ, so the assertion above has teeth.
    assert abs(xi_form / expected_z3) == pytest.approx(np.sqrt(2.0), rel=1e-12)


def test_no_slab_is_a_pure_resistance_divider():
    """With no material, the surface sits at T_o * R_si/(R_si+R_so) — at every frequency.

    This is the case that exposes the sign error in the paper's literal Eq. 11: that form gives
    T_o * (1 + R_so/(R_si+R_so)), which AMPLIFIES the driving signal. Physically the surface
    must sit between the two environments.
    """
    assembly = Assembly(layers=[], R_si=R_SI, R_so=R_SO)
    expected = R_SI / (R_SI + R_SO)

    for period in (3600.0, 86400.0, 86400.0 * 365):
        H = _H(assembly, period)
        assert H.real == pytest.approx(expected, rel=1e-12)
        assert H.imag == pytest.approx(0.0, abs=1e-12)
        assert abs(H) < 1.0, "a passive assembly can only damp, never amplify"


def test_steady_driving_gives_the_steady_state_divider():
    """A constant sol-air temperature must produce the steady-state surface temperature."""
    assembly = Assembly(layers=[Layer(thickness=0.2, **CONCRETE)], R_si=R_SI, R_so=R_SO)
    T_o, T_i = 30.0, 0.0
    T_driving = np.full(8760, T_o)

    T_so = solve_surface_temperature(T_driving, assembly, T_internal=T_i)

    U = assembly.steady_state_U()
    expected = T_o - U * (T_o - T_i) * R_SO
    assert np.allclose(T_so, expected, atol=1e-9)
    # Sanity on the value itself: a 200 mm slab is a poor insulator, so the surface sits only
    # slightly below the driving temperature.
    assert T_i < expected < T_o


def test_transfer_function_damps_at_every_frequency():
    """|H| < 1 for all harmonics and all real assemblies.

    The corrected transfer function damps; the paper's literal Eq. 11 amplifies. Any future
    refactor that reintroduces the paper's sign will trip this immediately.
    """
    for thickness in (0.02, 0.2, 1.0):
        assembly = Assembly(layers=[Layer(thickness=thickness, **CONCRETE)], R_si=R_SI, R_so=R_SO)
        for hours in (1, 3, 12, 24, 24 * 30, 8760):
            H = _H(assembly, hours * 3600.0)
            assert abs(H) < 1.0, f"|H|={abs(H)} at {hours} h, L={thickness}"


def test_thermally_thin_layer_tends_to_the_resistance_divider():
    """As the cyclic thickness tau -> 0 the slab stops storing heat and acts as pure resistance."""
    period = 86400.0
    thin = Assembly(layers=[Layer(thickness=1e-5, **CONCRETE)], R_si=R_SI, R_so=R_SO)

    R_layer = 1e-5 / CONCRETE["conductivity"]
    expected = (R_SI + R_layer) / (R_SI + R_layer + R_SO)
    assert _H(thin, period).real == pytest.approx(expected, rel=1e-6)


def test_thermally_thick_layer_tends_to_the_semi_infinite_admittance():
    """As tau -> infinity the inner boundary stops mattering and m1/m2 -> lambda*p.

    The semi-infinite solid is the classical closed-form limit of the transfer matrix, so this
    pins the layer matrix against textbook theory rather than against itself.
    """
    period = 86400.0
    thick = Assembly(layers=[Layer(thickness=5.0, **CONCRETE)], R_si=R_SI, R_so=R_SO)
    M = thick.total_matrix(period)
    admittance = M[0, 0] / M[0, 1]

    lam, rho, c = CONCRETE["conductivity"], CONCRETE["density"], CONCRETE["specific_heat"]
    p = np.sqrt(1j * (2.0 * np.pi / period) / (lam / (rho * c)))
    assert admittance == pytest.approx(lam * p, rel=1e-6)


def test_thicker_mass_damps_the_daily_swing_more():
    """Physical ordering: more exposed thermal mass -> smaller surface temperature swing."""
    n = 24 * 4
    T_driving = 10.0 * np.sin(2.0 * np.pi * np.arange(n) / 24.0)

    swings = []
    for thickness in (0.02, 0.10, 0.40):
        assembly = Assembly(layers=[Layer(thickness=thickness, **CONCRETE)], R_si=R_SI, R_so=R_SO)
        T_so = solve_surface_temperature(T_driving, assembly, T_internal=0.0)
        swings.append(T_so.max() - T_so.min())

    assert swings[0] > swings[1] > swings[2], f"mass should damp, got {swings}"


def test_u_value_matches_series_resistance():
    """The steady-state U-value is 1/sum(R), including both surface resistances."""
    layers = [Layer(thickness=0.1, **CONCRETE), Layer(thickness=0.05, conductivity=0.025,
                                                      density=30, specific_heat=1400)]
    assembly = Assembly(layers=layers, R_si=R_SI, R_so=R_SO)

    R = R_SI + R_SO + 0.1 / 1.40 + 0.05 / 0.025
    assert assembly.steady_state_U() == pytest.approx(1.0 / R, rel=1e-12)
    # Insulation dominates: U must be well under 0.5 W/m2K here.
    assert assembly.steady_state_U() < 0.5
