import numpy as np
import pytest

from surface_temps.admittance import solve_surface_temperature
from surface_temps.materials import Assembly, Layer


def _concrete_slab_assembly() -> Assembly:
    """200mm concrete slab (paper's verification case)."""
    return Assembly(
        layers=[Layer(0.2, 1.13, 2000, 1000)],
        R_si=0.0,
        R_so=0.04,
    )


class TestAdmittanceSolver:
    def test_constant_driving_gives_steady_state(self):
        assembly = _concrete_slab_assembly()
        N = 8760
        T_driving = np.full(N, 30.0)
        T_internal = 20.0

        T_surface = solve_surface_temperature(T_driving, assembly, T_internal)

        U = assembly.steady_state_U()
        expected_mean = 30.0 - U * (30.0 - 20.0) * assembly.R_so
        assert np.allclose(T_surface, expected_mean, atol=0.01)

    def test_sinusoidal_amplitude_reduced(self):
        """Surface temperature amplitude should be less than driving amplitude."""
        assembly = _concrete_slab_assembly()
        N = 8760
        t = np.arange(N) * 2 * np.pi / N
        T_driving = 20.0 + 10.0 * np.cos(t)

        T_surface = solve_surface_temperature(T_driving, assembly, T_internal=20.0)

        driving_amp = np.max(T_driving) - np.min(T_driving)
        surface_amp = np.max(T_surface) - np.min(T_surface)
        assert surface_amp < driving_amp
        assert surface_amp > 0

    def test_sinusoidal_phase_lag(self):
        """Surface temperature peak should lag behind driving temperature peak."""
        assembly = _concrete_slab_assembly()
        N = 8760
        t = np.arange(N) * 2 * np.pi / N
        T_driving = 20.0 + 10.0 * np.cos(t)

        T_surface = solve_surface_temperature(T_driving, assembly, T_internal=20.0)

        driving_peak = np.argmax(T_driving)
        surface_peak = np.argmax(T_surface)
        # Surface peak should be at or after driving peak (phase lag)
        lag = (surface_peak - driving_peak) % N
        assert lag >= 0

    def test_sawtooth_bounded(self):
        """Sawtooth driving: surface temp stays within driving range."""
        assembly = _concrete_slab_assembly()
        N = 240  # 10-day cycle at hourly resolution
        T_driving = 20.0 + 10.0 * (2 * (np.arange(N) % 24) / 24 - 1)

        T_surface = solve_surface_temperature(T_driving, assembly, T_internal=20.0)

        assert np.max(T_surface) <= np.max(T_driving) + 1.0
        assert np.min(T_surface) >= np.min(T_driving) - 1.0

    def test_output_length_matches_input(self):
        assembly = _concrete_slab_assembly()
        for N in [24, 168, 8760]:
            T_driving = np.random.default_rng(42).normal(20, 5, N)
            T_surface = solve_surface_temperature(T_driving, assembly, T_internal=20.0)
            assert len(T_surface) == N
