from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from surface_temps.admittance import solve_surface_temperature
from surface_temps.convection import h_convective
from surface_temps.geometry import NeighborhoodGeometry
from surface_temps.materials import Assembly
from surface_temps.solar import sky_temperature, sol_air_temperature, transpose_irradiance
from surface_temps.weather import WeatherData


@dataclass
class Surface:
    name: str
    assembly: Assembly
    tilt: float  # 0 = horizontal, 90 = vertical
    azimuth: float  # degrees from north
    absorptivity: float
    emissivity: float
    T_internal: float  # degC, deep ground or indoor temp
    face_vertices: np.ndarray | None = None  # for 3D plotting linkage


def build_surfaces(
    geometry: NeighborhoodGeometry,
    weather: WeatherData,
) -> list[Surface]:
    """Convert geometry into Surface objects ready for simulation."""
    T_deep = weather.annual_mean_temp
    T_indoor = 22.0
    surfaces = []

    for box in geometry.boxes:
        for face in box.faces():
            if face.tilt == 0:
                # Roof
                surfaces.append(
                    Surface(
                        name=face.name,
                        assembly=box.roof_assembly,
                        tilt=face.tilt,
                        azimuth=face.azimuth,
                        absorptivity=box.roof_absorptivity,
                        emissivity=box.roof_emissivity,
                        T_internal=T_indoor,
                        face_vertices=face.vertices,
                    )
                )
            else:
                # Wall
                surfaces.append(
                    Surface(
                        name=face.name,
                        assembly=box.wall_assembly,
                        tilt=face.tilt,
                        azimuth=face.azimuth,
                        absorptivity=box.wall_absorptivity,
                        emissivity=box.wall_emissivity,
                        T_internal=T_indoor,
                        face_vertices=face.vertices,
                    )
                )

    for patch in geometry.ground_patches:
        surfaces.append(
            Surface(
                name=patch.name,
                assembly=patch.assembly,
                tilt=0,
                azimuth=0,
                absorptivity=patch.absorptivity,
                emissivity=patch.emissivity,
                T_internal=T_deep,
                face_vertices=patch.vertices,
            )
        )

    return surfaces


def simulate_all(
    surfaces: list[Surface],
    weather: WeatherData,
) -> dict[str, np.ndarray]:
    """Run admittance solver for every surface. Returns {name: T_surface(8760)}."""
    T_sky = sky_temperature(weather)
    h_c = h_convective(weather.wind_speed)
    h_r = 5.0  # W/m2-K, linearized radiative coefficient

    # Cache transposed irradiance by (tilt, azimuth) to avoid redundant pvlib calls
    irradiance_cache: dict[tuple[float, float], np.ndarray] = {}
    results = {}

    for surf in surfaces:
        key = (surf.tilt, round(surf.azimuth, 1))
        if key not in irradiance_cache:
            irradiance_cache[key] = transpose_irradiance(weather, surf.tilt, surf.azimuth)
        Q_sol = irradiance_cache[key]

        if surf.tilt == 0:
            T_radiant = T_sky
        else:
            T_radiant = weather.temp_air

        T_sol = sol_air_temperature(
            weather.temp_air, T_radiant, Q_sol, surf.absorptivity, h_c, h_r
        )

        T_surface = solve_surface_temperature(T_sol, surf.assembly, surf.T_internal)
        results[surf.name] = T_surface

    return results
