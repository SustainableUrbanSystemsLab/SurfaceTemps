from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

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
    max_iterations: int = 200,
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
    with ``q_co = dh*(T_e - T_s)``.

    ``q_co`` depends on the surface temperature, so the paper solves it by iteration and states
    that fewer than five passes suffice. That does NOT hold for the wind series in a real EPW:
    measured across the whole material library on Atlanta TMY3, the Picard gain exceeds 1 for
    every single material, so plain iteration diverges and even a damped version had not reached
    1e-4 after fifty passes for the metal roofs. The relation is linear in ``T_s``, so this
    solves it directly instead (see below) — exact, and independent of how variable the wind is.

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

    # This is a LINEAR system, so solve it as one rather than iterating.
    #
    # Simple Picard iteration on q_co has gain |delta_h|/h_bar * |H|, and on real weather that
    # exceeds 1 for EVERY material in the library (h_e spans ~10 to ~70 W/m2-K about a mean near
    # 26, and |H| approaches 1 for any thin or insulated outer layer). Undamped it diverges —
    # an insulated roof reached 3e4 degC. Damped it converges, but slowly and unpredictably:
    # metal roofs still had not reached 1e-4 after fifty passes, and the loop then returned a
    # silently unconverged answer.
    #
    # Writing A for the (affine) admittance solve and D for multiplication by delta_h/h_bar:
    #
    #     T_s = A[u - D T_s]        with u = T_env + (Q + delta_h*T_env)/h_bar
    #     (I + L·D) T_s = L[u] + c   where A[x] = L[x] + c, L linear
    #
    # L costs one FFT pair, so a Krylov solve converges in tens of matvecs regardless of gain
    # and needs no relaxation parameter.
    T_env = np.asarray(T_environmental, dtype=float)
    Q_abs = np.asarray(Q_absorbed, dtype=float)
    n = T_env.size
    scale = delta_h / h_bar

    # Split the affine solve into its linear part and its constant offset.
    zero = np.zeros(n)
    offset = solve_surface_temperature(zero, effective, T_internal)  # = A[0] = c

    def linear(x: np.ndarray) -> np.ndarray:
        return solve_surface_temperature(x, effective, T_internal) - offset

    rhs = linear(T_env + (Q_abs + delta_h * T_env) / h_bar) + offset

    def matvec(x: np.ndarray) -> np.ndarray:
        return x + linear(scale * x)

    operator = LinearOperator((n, n), matvec=matvec, dtype=float)
    guess = solve_surface_temperature(T_env + Q_abs / h_bar, effective, T_internal)

    # gmres measures the residual in the 2-NORM, but the tolerance we care about is per-hour
    # (max-norm) in kelvin. Over 8760 samples the two differ by up to sqrt(n), so ask for the
    # tighter 2-norm bound that guarantees the max-norm one.
    rhs_norm = float(np.linalg.norm(rhs))
    solution, info = gmres(
        operator, rhs, x0=guess,
        rtol=tol / max(rhs_norm, 1.0), atol=0.0,
        restart=min(80, n), maxiter=max_iterations,
    )

    if info != 0:
        raise RuntimeError(
            f"variable-h correction did not converge (gmres info={info}). The hourly h_e series "
            "may be extreme; fall back to solve_surface_temperature with a fixed R_so."
        )

    # Verify the fixed point actually holds rather than trusting the solver's own flag.
    residual = float(np.max(np.abs(matvec(solution) - rhs)))
    if not np.isfinite(residual) or residual > max(1e-6, tol):
        raise RuntimeError(
            f"variable-h correction converged to a residual of {residual:.3g} K, above tolerance"
        )

    return solution


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
