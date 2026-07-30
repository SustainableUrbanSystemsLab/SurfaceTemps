from __future__ import annotations

from dataclasses import replace

import numpy as np

from surface_temps.materials import Assembly

_TRANSFER_FUNCTION_CACHE: dict[tuple, tuple[np.ndarray, float]] = {}


def solve_surface_temperature(
    T_driving: np.ndarray,
    assembly: Assembly,
    T_internal: float = 0.0,
) -> np.ndarray:
    """Compute surface temperature using the frequency-domain admittance method.

    The transfer matrix (R_si + layers, excluding R_so) maps from the external
    surface to the internal environment. With constant internal temperature
    (T_i_fluct = 0), the external surface heat flux is q_so = -(m1/m2)*T_so
    (positive outward). Combined with q_so = (T_so - T_o)/R_so, the surface
    temperature transfer function is H_n = 1/(1 + m1_n*R_so/m2_n).
    """
    N = len(T_driving)
    dt = 3600.0

    T_mean = np.mean(T_driving)
    T_fluct = T_driving - T_mean

    T_fft = np.fft.rfft(T_fluct)

    R_so = assembly.R_so
    H, U = _cached_transfer_function(assembly, N, dt)
    T_so_mean = T_mean - U * (T_mean - T_internal) * R_so

    T_so_fft = T_fft * H

    T_so_fluct = np.fft.irfft(T_so_fft, n=N)
    return T_so_mean + T_so_fluct


def solve_surface_temperature_variable_h(
    T_environmental: np.ndarray,
    Q_absorbed: np.ndarray,
    h_e: np.ndarray,
    assembly: Assembly,
    T_internal: float = 0.0,
    max_iterations: int = 20,
    tol: float = 1e-4,
) -> np.ndarray:
    """Surface temperature with an HOURLY surface heat transfer coefficient (paper Eq. 20-22).

    The plain solver assumes one constant external resistance R_so. That is a real limitation,
    not a cosmetic one: the sol-air temperature divides the absorbed solar by the HOURLY
    ``h_e``, while the solver couples the surface to it through a FIXED ``1/R_so``. On a calm
    sunny afternoon (h_e ~ 10.7) against the default R_so = 0.04 (h_e = 25) that injects the
    absorbed solar roughly 2.3x too strongly; on a windy hour it under-injects.

    The paper's remedy (Eq. 20-22) keeps the constant-resistance machinery and moves the
    difference into a fictitious corrective flux. Writing the surface balance with
    ``h_e(t) = h_bar + dh(t)``::

        h_e(t)*(T_e - T_s) + a*Q = h_bar*(T_e - T_s) + dh*(T_e - T_s) + a*Q

    so with ``R_so = 1/h_bar`` the driving temperature becomes ``T_sol = T_e + (a*Q + q_co)/h_bar``
    with ``q_co = dh*(T_e - T_s)``. Since ``q_co`` needs the surface temperature it is solved by
    iteration; the paper notes fewer than five passes suffice, which matches what we see.

    Args:
        T_environmental: combined air/radiant environmental temperature (degC), i.e.
            ``(h_c*T_air + h_r*T_radiant)/h_e``.
        Q_absorbed: absorbed solar flux ``alpha * Q_sol`` (W/m2).
        h_e: hourly combined surface coefficient ``h_c + h_r`` (W/m2-K).
        assembly: the build-up. Its ``R_so`` is REPLACED by ``1/mean(h_e)``.
    """
    h_e = np.asarray(h_e, dtype=float)
    if np.any(h_e <= 0):
        raise ValueError("h_e must be positive everywhere")

    h_bar = float(np.mean(h_e))
    effective = replace(assembly, R_so=1.0 / h_bar)
    delta_h = h_e - h_bar

    # The undamped fixed point has gain |delta_h|/h_bar, which for a realistic wind series
    # exceeds 1 (h_e spans ~10 to ~70 W/m2-K about a mean near 26). For a surface that closely
    # follows its driving temperature — a thin or well-insulated outer layer, |H| ~ 1 — plain
    # iteration then DIVERGES, and spectacularly: an insulated roof ran away to 3e4 degC before
    # this damping was added. Under-relaxation with adaptive backoff is what makes it robust.
    weight = 0.7

    T_sol = T_environmental + Q_absorbed / h_bar
    T_surface = solve_surface_temperature(T_sol, effective, T_internal)
    previous_shift = np.inf

    for _ in range(max_iterations):
        q_co = delta_h * (T_environmental - T_surface)
        T_sol = T_environmental + (Q_absorbed + q_co) / h_bar
        candidate = solve_surface_temperature(T_sol, effective, T_internal)

        updated = T_surface + weight * (candidate - T_surface)
        shift = float(np.max(np.abs(updated - T_surface)))

        if shift > previous_shift:
            # Diverging: back off and retry from where we were rather than accepting it.
            weight *= 0.5
            if weight < 1e-3:
                raise RuntimeError(
                    "variable-h correction failed to converge; h_e is too variable for this "
                    "assembly. Fall back to solve_surface_temperature with a fixed R_so."
                )
            previous_shift = np.inf
            continue

        T_surface = updated
        previous_shift = shift
        if shift < tol:
            break

    return T_surface


def _cached_transfer_function(
    assembly: Assembly,
    n_samples: int,
    dt: float,
) -> tuple[np.ndarray, float]:
    key = (_assembly_key(assembly), n_samples, dt)
    if key not in _TRANSFER_FUNCTION_CACHE:
        _TRANSFER_FUNCTION_CACHE[key] = _compute_transfer_function(
            assembly,
            n_samples,
            dt,
        )
    return _TRANSFER_FUNCTION_CACHE[key]


def _compute_transfer_function(
    assembly: Assembly,
    n_samples: int,
    dt: float,
) -> tuple[np.ndarray, float]:
    total_period = n_samples * dt
    R_so = assembly.R_so
    U = assembly.steady_state_U()
    H = np.zeros(n_samples // 2 + 1, dtype=complex)

    for n in range(1, len(H)):
        period_n = total_period / n
        M = assembly.total_matrix(period_n)
        m1, m2 = M[0, 0], M[0, 1]
        H[n] = 1.0 / (1.0 + m1 * R_so / m2)

    return H, U


def _assembly_key(assembly: Assembly) -> tuple:
    return (
        assembly.R_si,
        assembly.R_so,
        tuple(
            (
                layer.thickness,
                layer.conductivity,
                layer.density,
                layer.specific_heat,
            )
            for layer in assembly.layers
        ),
    )
