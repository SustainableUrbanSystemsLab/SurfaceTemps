from __future__ import annotations

import numpy as np
import pvlib

from surface_temps.constants import STEFAN_BOLTZMANN
from surface_temps.weather import WeatherData


def transpose_irradiance(
    weather: WeatherData, tilt: float, azimuth: float
) -> np.ndarray:
    """Plane-of-array irradiance for a surface at given tilt and azimuth (W/m2)."""
    solar_pos = weather.solar_position

    result = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solar_pos["apparent_zenith"],
        solar_azimuth=solar_pos["azimuth"],
        dni=weather.dni,
        ghi=weather.ghi,
        dhi=weather.dhi,
        model="isotropic",
    )

    poa = result["poa_global"].values
    poa = np.nan_to_num(poa, nan=0.0)
    return np.maximum(poa, 0.0)


def sky_temperature(weather: WeatherData, emissivity: float = 0.90) -> np.ndarray:
    """Sky temperature (degC) from EPW horizontal infrared radiation."""
    ir = weather.infrared_horizontal
    valid = ir > 0
    T_sky_K = np.full_like(ir, dtype=float, fill_value=273.15)
    T_sky_K[valid] = (ir[valid] / (emissivity * STEFAN_BOLTZMANN)) ** 0.25
    return T_sky_K - 273.15


def sol_air_temperature(
    T_air: np.ndarray,
    T_radiant: np.ndarray,
    Q_sol: np.ndarray,
    alpha: float,
    h_c: np.ndarray,
    h_r: float | np.ndarray = 5.0,
) -> np.ndarray:
    """Sol-air temperature (degC), Eq. 1 from Beckett et al."""
    h_e = h_c + h_r
    return (h_c * T_air + h_r * T_radiant + alpha * Q_sol) / h_e
