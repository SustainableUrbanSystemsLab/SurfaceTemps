from __future__ import annotations

import numpy as np
import pvlib

from surface_temps.constants import STEFAN_BOLTZMANN
from surface_temps.weather import WeatherData


def transpose_irradiance(weather: WeatherData, tilt: float, azimuth: float) -> np.ndarray:
    """Plane-of-array irradiance for a surface at given tilt and azimuth (W/m2)."""
    return transpose_irradiance_components(weather, tilt, azimuth)["poa_global"]


def transpose_irradiance_components(
    weather: WeatherData, tilt: float, azimuth: float
) -> dict[str, np.ndarray]:
    """Plane-of-array irradiance components for a surface orientation (W/m2)."""
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

    components = {}
    for name in [
        "poa_global",
        "poa_direct",
        "poa_diffuse",
        "poa_sky_diffuse",
        "poa_ground_diffuse",
    ]:
        values = result[name].values
        values = np.nan_to_num(values, nan=0.0)
        components[name] = np.maximum(values, 0.0)
    return components


def sun_direction_vectors(weather: WeatherData) -> np.ndarray:
    """Unit vectors from each surface point toward the sun in x=east, y=north, z=up."""
    solar_pos = weather.solar_position
    zenith = np.radians(solar_pos["apparent_zenith"].values)
    azimuth = np.radians(solar_pos["azimuth"].values)
    sin_zenith = np.sin(zenith)

    vectors = np.column_stack(
        [
            sin_zenith * np.sin(azimuth),
            sin_zenith * np.cos(azimuth),
            np.cos(zenith),
        ]
    )
    vectors = np.nan_to_num(vectors, nan=0.0)
    return vectors


def sky_temperature(weather: WeatherData) -> np.ndarray:
    """Effective sky temperature (degC) from EPW horizontal infrared radiation.

    The EPW ``Horizontal Infrared Radiation Intensity`` field IS the downwelling long-wave
    flux from the sky, so the effective sky temperature is defined by ``IR = sigma * T_sky^4``
    with no emissivity term: the sky's emissivity is already folded into the measured flux.

    This previously divided by an emissivity of 0.90, which inflated T_sky by
    ``(1/0.9)^0.25 = 1.0267`` in KELVIN — about +7.5 K for Atlanta TMY3. That left the sky only
    ~2.9 K below air temperature, where the real clear-sky depression is ~10 K, and so
    systematically under-predicted radiative cooling and over-predicted surface temperatures.
    The surface emissivity belongs in the radiative coefficient (see :func:`h_radiative`), not
    here.
    """
    ir = weather.infrared_horizontal
    valid = ir > 0
    T_sky_K = np.full_like(ir, dtype=float, fill_value=273.15)
    T_sky_K[valid] = (ir[valid] / STEFAN_BOLTZMANN) ** 0.25
    return T_sky_K - 273.15


def h_radiative(
    emissivity: float,
    T_reference: np.ndarray | float,
) -> np.ndarray:
    """Linearised long-wave radiative coefficient h_r = 4 * eps * sigma * T^3 (W/m2-K).

    ``T_reference`` is in degC and is the temperature about which the fourth-power law is
    linearised; the hourly air temperature is a reasonable choice outdoors.

    Using a fixed h_r = 5.0 is only right for high-emissivity surfaces: 5.0 corresponds to
    eps ~ 0.90 at 290 K. Bare and coated metals have emissivities as low as 0.05-0.2, where the
    true h_r is 0.3-1.1 W/m2-K — a constant 5.0 overstates their radiative coupling by up to
    twentyfold and would make a metal roof track sky temperature far too strongly.
    """
    T_K = np.asarray(T_reference, dtype=float) + 273.15
    return 4.0 * emissivity * STEFAN_BOLTZMANN * T_K**3


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
