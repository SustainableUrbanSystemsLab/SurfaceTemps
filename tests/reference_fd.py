"""Independent 1D finite-difference slab solver, used only to validate the admittance method.

This deliberately shares NO code with ``surface_temps``. It integrates the heat equation in
the TIME domain on a node network, where the frequency-domain solver works analytically in
the FREQUENCY domain, so agreement between them is real evidence rather than a tautology.
It is the check the paper itself performs in its Figure 3 (Fourier method vs FEA under a
sawtooth driving temperature).

Node layout, for a build-up of one or more layers::

    T_o ──[R_so]── T₀ ── T₁ ── … ── T_N ──[R_si]── T_i
                    │     │           │
                 half-C  full-C    half-C

``T₀`` is the outer surface node, so it is directly comparable to what
``solve_surface_temperature`` returns. End nodes carry half a cell of capacitance; interior
nodes carry a full cell. At a material interface the two half-cells contribute their own
``ρc`` and the conductance across the joint is the series (harmonic) combination — the
standard control-volume treatment.

Time integration is Crank-Nicolson (second-order, unconditionally stable). The driving series
is looped until the slab reaches a periodic steady state, which removes any dependence on the
initial condition — the same property the frequency-domain method gets for free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FdLayer:
    """One material layer. Mirrors ``surface_temps.materials.Layer`` but is independent of it."""

    thickness: float  # m
    conductivity: float  # W/m-K
    density: float  # kg/m3
    specific_heat: float  # J/kg-K
    n_cells: int = 40


def _build_network(
    layers: list[FdLayer],
) -> tuple[np.ndarray, np.ndarray]:
    """Node capacitances (J/m2-K) and inter-node conductances (W/m2-K), OUTER to INNER.

    Every cell hands half its heat capacity to each of the two nodes bounding it, so an
    interface node naturally ends up with half a cell from each adjoining material. The
    conductance between two nodes is that of the single cell spanning them.
    """
    n_nodes = 1 + sum(layer.n_cells for layer in layers)
    cap = np.zeros(n_nodes)
    cond = np.zeros(n_nodes - 1)

    node = 0
    for layer in layers:
        dx = layer.thickness / layer.n_cells
        cell_cap = layer.density * layer.specific_heat * dx
        cell_cond = layer.conductivity / dx
        for _ in range(layer.n_cells):
            cond[node] = cell_cond
            cap[node] += cell_cap / 2.0
            cap[node + 1] += cell_cap / 2.0
            node += 1

    return cap, cond


def solve_fd(
    T_driving: np.ndarray,
    layers: list[FdLayer],
    R_so: float,
    R_si: float,
    T_internal: float = 0.0,
    dt: float = 3600.0,
    substeps: int = 60,
    n_cycles: int = 12,
    tol: float = 1e-4,
    T_env_fn=None,
    h_e_series: np.ndarray | None = None,
    q_absorbed_series: np.ndarray | None = None,
) -> np.ndarray:
    """Outer surface temperature for a driving series, by time-domain integration.

    ``T_driving`` is the sol-air temperature at ``dt`` spacing. The series is treated as one
    period and replayed until the response repeats to within ``tol``, so the answer is the
    periodic steady state and carries no memory of the initial condition.

    Returns the outer-surface temperature over the final cycle, aligned with ``T_driving``.
    """
    cap, cond = _build_network(layers)
    n_nodes = cap.size
    if cond.size != n_nodes - 1:
        raise AssertionError(f"network mismatch: {n_nodes} nodes, {cond.size} conductances")

    n_steps = len(T_driving)
    h = dt / substeps

    # Conductance to the environment at each end.
    g_out = 1.0 / R_so
    g_in = np.inf if R_si == 0.0 else 1.0 / R_si

    # Assemble the constant conductance matrix K (W/m2-K) where dT/dt = C^-1 (K T + b).
    K = np.zeros((n_nodes, n_nodes))
    for j in range(n_nodes - 1):
        K[j, j] -= cond[j]
        K[j, j + 1] += cond[j]
        K[j + 1, j] += cond[j]
        K[j + 1, j + 1] -= cond[j]
    K[0, 0] -= g_out
    if np.isfinite(g_in):
        K[-1, -1] -= g_in

    C = np.diag(cap)

    # With an hourly h_e the outer conductance changes each hour, so the Crank-Nicolson
    # matrices are rebuilt per hour (and cached, since they repeat every cycle). This applies
    # the PHYSICAL boundary condition h_e(t)*(T_env - T_s) + q_absorbed directly, which is what
    # makes it an independent check of the paper's Eq. 20-22 correction rather than a
    # re-derivation of it.
    variable_h = h_e_series is not None
    if variable_h:
        h_e_series = np.asarray(h_e_series, dtype=float)
        if q_absorbed_series is None:
            q_absorbed_series = np.zeros_like(h_e_series)
        q_absorbed_series = np.asarray(q_absorbed_series, dtype=float)

    def matrices_for(g_out_value: float):
        K_local = K.copy()
        K_local[0, 0] += g_out - g_out_value  # swap the default conductance for this one
        A_local = C / h - K_local / 2.0
        return np.linalg.inv(A_local), C / h + K_local / 2.0

    # Crank-Nicolson: (C/h - K/2) T^{n+1} = (C/h + K/2) T^n + (b^n + b^{n+1})/2
    A = C / h - K / 2.0
    B = C / h + K / 2.0
    A_inv = np.linalg.inv(A)
    _hour_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    T = np.full(n_nodes, float(np.mean(T_driving)))
    if np.isfinite(g_in):
        T[-1] = T_internal

    def source(T_env: float, g_outer: float = None, q_extra: float = 0.0) -> np.ndarray:
        b = np.zeros(n_nodes)
        b[0] += (g_out if g_outer is None else g_outer) * T_env + q_extra
        if np.isfinite(g_in):
            b[-1] += g_in * T_internal
        return b

    # Sub-hour driving temperature. By default the hourly samples are interpolated linearly,
    # which is what a time-domain tool would do. Note that the frequency-domain method instead
    # reconstructs the signal trigonometrically between samples, so for a signal that is not
    # band-limited the two solvers are genuinely driven by different sub-hour inputs and will
    # differ by a small amount that does NOT vanish under grid refinement. Passing an exact
    # ``T_env_fn(t_hours)`` removes that difference and lets the comparison converge properly.
    def T_env_at(step: int, frac: float) -> float:
        if T_env_fn is not None:
            return float(T_env_fn((step + frac) * dt / 3600.0))
        a = T_driving[step % n_steps]
        b = T_driving[(step + 1) % n_steps]
        return a + (b - a) * frac

    surface = np.zeros(n_steps)
    previous = None
    for cycle in range(n_cycles):
        for step in range(n_steps):
            surface[step] = T[0]
            idx = step % n_steps
            if variable_h:
                if idx not in _hour_cache:
                    _hour_cache[idx] = matrices_for(h_e_series[idx])
                A_step, B_step = _hour_cache[idx]
                extra = q_absorbed_series[idx]
                g_step = h_e_series[idx]
            else:
                A_step, B_step = A_inv, B
                extra = 0.0
                g_step = g_out

            for s in range(substeps):
                env_now = T_env_at(step, s / substeps)
                env_next = T_env_at(step, (s + 1) / substeps)
                b_now = source(env_now, g_step, extra)
                b_next = source(env_next, g_step, extra)
                rhs = B_step @ T + (b_now + b_next) / 2.0
                T = A_step @ rhs
        if previous is not None and np.max(np.abs(surface - previous)) < tol:
            break
        previous = surface.copy()

    return surface


def sawtooth(n: int, amplitude: float = 1.0, period_steps: int = 24) -> np.ndarray:
    """Sawtooth driving temperature, matching the paper's Figure 3 excitation."""
    t = np.arange(n) % period_steps
    return amplitude * (2.0 * np.abs(2.0 * (t / period_steps) - 1.0) - 1.0)
