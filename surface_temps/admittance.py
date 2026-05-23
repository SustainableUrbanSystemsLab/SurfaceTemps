from __future__ import annotations

import numpy as np

from surface_temps.materials import Assembly


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
    total_period = N * dt

    T_mean = np.mean(T_driving)
    T_fluct = T_driving - T_mean

    T_fft = np.fft.rfft(T_fluct)

    R_so = assembly.R_so
    U = assembly.steady_state_U()
    T_so_mean = T_mean - U * (T_mean - T_internal) * R_so

    T_so_fft = np.zeros_like(T_fft)
    T_so_fft[0] = 0.0

    for n in range(1, len(T_fft)):
        period_n = total_period / n
        M = assembly.total_matrix(period_n)
        m1, m2 = M[0, 0], M[0, 1]
        H_n = 1.0 / (1.0 + m1 * R_so / m2)
        T_so_fft[n] = T_fft[n] * H_n

    T_so_fluct = np.fft.irfft(T_so_fft, n=N)
    return T_so_mean + T_so_fluct
