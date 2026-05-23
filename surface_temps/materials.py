from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from surface_temps.constants import MATERIALS, R_SI_GROUND, R_SI_WALL, R_SO


@dataclass
class Layer:
    thickness: float  # m
    conductivity: float  # W/m-K
    density: float  # kg/m3
    specific_heat: float  # J/kg-K

    @classmethod
    def from_name(cls, name: str, thickness: float) -> Layer:
        props = MATERIALS[name]
        return cls(
            thickness=thickness,
            conductivity=props["conductivity"],
            density=props["density"],
            specific_heat=props["specific_heat"],
        )

    def transfer_matrix(self, period: float) -> np.ndarray:
        """2x2 complex transfer matrix for this layer at the given period (seconds).

        Uses the standard diffusion equation convention: z2 = sinh(pL)/(λp),
        z3 = λp·sinh(pL), where p = sqrt(iω/α) and α = λ/(ρc).
        """
        lam = self.conductivity
        a = lam / (self.density * self.specific_heat)  # thermal diffusivity
        omega = 2.0 * np.pi / period
        p = np.sqrt(1j * omega / a)
        pL = p * self.thickness

        z1 = np.cosh(pL)
        z2 = np.sinh(pL) / (lam * p)
        z3 = lam * p * np.sinh(pL)
        z4 = z1

        return np.array([[z1, z2], [z3, z4]])

    @property
    def thermal_resistance(self) -> float:
        return self.thickness / self.conductivity


@dataclass
class Assembly:
    layers: list[Layer] = field(default_factory=list)
    R_si: float = 0.0  # m2-K/W, internal surface resistance
    R_so: float = R_SO  # m2-K/W, external surface resistance

    def total_matrix(self, period: float) -> np.ndarray:
        """Transfer matrix for internal surface resistance + layers (excludes R_so).

        R_so is excluded because the admittance solver applies it as a boundary
        condition to extract the external surface temperature.
        """
        M_si = np.array([[1, self.R_si], [0, 1]], dtype=complex)

        M = M_si.copy()
        for layer in self.layers:
            M = M @ layer.transfer_matrix(period)

        return M

    def steady_state_U(self) -> float:
        """Steady-state U-value (W/m2-K)."""
        R_total = self.R_si + self.R_so
        for layer in self.layers:
            R_total += layer.thermal_resistance
        return 1.0 / R_total


def concrete_ground() -> Assembly:
    return Assembly(
        layers=[
            Layer.from_name("soil_subsoil", 1.0),
            Layer.from_name("concrete", 0.2),
        ],
        R_si=R_SI_GROUND,
    )


def brick_ground() -> Assembly:
    return Assembly(
        layers=[
            Layer.from_name("soil_subsoil", 1.0),
            Layer.from_name("sand", 0.1),
            Layer.from_name("brick", 0.065),
        ],
        R_si=R_SI_GROUND,
    )


def grass_ground() -> Assembly:
    return Assembly(
        layers=[
            Layer.from_name("soil_subsoil", 1.0),
            Layer.from_name("soil_topsoil", 0.3),
            Layer.from_name("sod", 0.05),
        ],
        R_si=R_SI_GROUND,
    )


def brick_wall() -> Assembly:
    return Assembly(
        layers=[
            Layer.from_name("dense_plaster", 0.013),
            Layer.from_name("concrete_block", 0.150),
            Layer.from_name("air_gap_25mm", 0.025),
            Layer.from_name("brick", 0.102),
        ],
        R_si=R_SI_WALL,
    )


def concrete_roof() -> Assembly:
    return Assembly(
        layers=[
            Layer.from_name("dense_plaster", 0.013),
            Layer.from_name("cast_concrete", 0.150),
            Layer.from_name("polyurethane_insulation", 0.035),
            Layer.from_name("screed", 0.015),
            Layer.from_name("waterproof_membrane", 0.001),
        ],
        R_si=R_SI_WALL,
    )
