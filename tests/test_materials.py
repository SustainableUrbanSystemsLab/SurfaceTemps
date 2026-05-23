import numpy as np
import pytest

from surface_temps.materials import Assembly, Layer, concrete_ground, brick_wall


class TestLayer:
    def test_transfer_matrix_determinant_is_one(self):
        """det(Z) = z1*z4 - z2*z3 = 1 for a single layer."""
        layer = Layer(thickness=0.2, conductivity=1.13, density=2000, specific_heat=1000)
        for period in [3600, 86400, 86400 * 365]:
            M = layer.transfer_matrix(period)
            det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
            assert abs(det - 1.0) < 1e-8, f"det={det} at period={period}"

    def test_from_name(self):
        layer = Layer.from_name("concrete", 0.2)
        assert layer.conductivity == 1.13
        assert layer.thickness == 0.2

    def test_thermal_resistance(self):
        layer = Layer(thickness=0.2, conductivity=1.0, density=2000, specific_heat=1000)
        assert layer.thermal_resistance == pytest.approx(0.2)


class TestAssembly:
    def test_steady_state_U(self):
        assembly = Assembly(
            layers=[Layer(0.2, 1.0, 2000, 1000)],
            R_si=0.13,
            R_so=0.04,
        )
        expected_U = 1.0 / (0.13 + 0.2 + 0.04)
        assert assembly.steady_state_U() == pytest.approx(expected_U)

    def test_total_matrix_determinant(self):
        assembly = concrete_ground()
        for period in [86400, 86400 * 7, 86400 * 365]:
            M = assembly.total_matrix(period)
            det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
            assert abs(det - 1.0) < 1e-8, f"det={det} at period={period}"

    def test_factory_functions_produce_valid_assemblies(self):
        for factory in [concrete_ground, brick_wall]:
            assembly = factory()
            assert len(assembly.layers) > 0
            assert assembly.steady_state_U() > 0
