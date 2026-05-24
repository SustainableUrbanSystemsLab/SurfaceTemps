from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from surface_temps.admittance import solve_surface_temperature
from surface_temps.convection import h_convective
from surface_temps.geometry import NeighborhoodGeometry
from surface_temps.materials import Assembly
from surface_temps.radiation import (
    build_radiation_context,
    ideal_ground_view_factor,
    ideal_sky_view_factor,
    sunlit_factors,
)
from surface_temps.solar import (
    sky_temperature,
    sol_air_temperature,
    transpose_irradiance_components,
)
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
    surface_type: str = "surface"
    view_group: str = ""


def build_surfaces(
    geometry: NeighborhoodGeometry,
    weather: WeatherData,
) -> list[Surface]:
    """Convert geometry into Surface objects ready for simulation."""
    T_deep = weather.annual_mean_temp
    T_indoor = 22.0
    surfaces = []

    for mesh_surface in geometry.mesh_surfaces:
        T_internal = T_deep if mesh_surface.boundary == "ground" else T_indoor
        surfaces.append(
            Surface(
                name=mesh_surface.name,
                assembly=mesh_surface.assembly,
                tilt=mesh_surface.tilt,
                azimuth=mesh_surface.azimuth,
                absorptivity=mesh_surface.absorptivity,
                emissivity=mesh_surface.emissivity,
                T_internal=T_internal,
                face_vertices=mesh_surface.vertices,
                surface_type=mesh_surface.surface_type,
                view_group=mesh_surface.view_group,
            )
        )

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
                        surface_type="roof",
                        view_group=face.name,
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
                        surface_type="wall",
                        view_group=face.name,
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
                surface_type="ground",
                view_group=patch.name,
            )
        )

    return surfaces


def simulate_all(
    surfaces: list[Surface],
    weather: WeatherData,
    *,
    use_occlusion: bool = True,
    use_view_factors: bool = True,
) -> dict[str, np.ndarray]:
    """Run admittance solver for every surface. Returns {name: T_surface(8760)}."""
    T_sky = sky_temperature(weather)
    h_c = h_convective(weather.wind_speed)
    h_r = 5.0  # W/m2-K, linearized radiative coefficient

    radiation_context = build_radiation_context(
        surfaces,
        compute_view_factors=use_view_factors,
    ) if (use_occlusion or use_view_factors) else None

    # Cache transposed irradiance by (tilt, azimuth) to avoid redundant pvlib calls.
    irradiance_cache: dict[tuple[float, float], dict[str, np.ndarray]] = {}
    results = {}

    for surface_index, surf in enumerate(surfaces):
        key = (surf.tilt, round(surf.azimuth, 1))
        if key not in irradiance_cache:
            irradiance_cache[key] = transpose_irradiance_components(
                weather, surf.tilt, surf.azimuth
            )
        components = irradiance_cache[key]
        Q_sol = _surface_irradiance(
            surf,
            surface_index,
            weather,
            components,
            radiation_context,
            use_occlusion=use_occlusion,
            use_view_factors=use_view_factors,
        )

        if radiation_context is not None and use_view_factors:
            sky_view = min(
                radiation_context.sky_view[surface_index],
                ideal_sky_view_factor(surf.tilt),
            )
            T_radiant = sky_view * T_sky + (1.0 - sky_view) * weather.temp_air
        elif surf.tilt == 0:
            T_radiant = T_sky
        else:
            T_radiant = weather.temp_air

        T_sol = sol_air_temperature(
            weather.temp_air, T_radiant, Q_sol, surf.absorptivity, h_c, h_r
        )

        T_surface = solve_surface_temperature(T_sol, surf.assembly, surf.T_internal)
        results[surf.name] = T_surface

    return results


def _surface_irradiance(
    surf: Surface,
    surface_index: int,
    weather: WeatherData,
    components: dict[str, np.ndarray],
    radiation_context,
    *,
    use_occlusion: bool,
    use_view_factors: bool,
) -> np.ndarray:
    if radiation_context is None:
        return components["poa_global"]

    direct_factor = (
        sunlit_factors(surface_index, surf, weather, radiation_context)
        if use_occlusion
        else 1.0
    )
    direct = components["poa_direct"] * direct_factor

    if not use_view_factors:
        return direct + components["poa_diffuse"]

    ideal_sky = ideal_sky_view_factor(surf.tilt)
    ideal_ground = ideal_ground_view_factor(surf.tilt)

    sky_scale = _view_scale(radiation_context.sky_view[surface_index], ideal_sky)
    ground_scale = _view_scale(
        radiation_context.ground_view[surface_index], ideal_ground
    )

    return (
        direct
        + components["poa_sky_diffuse"] * sky_scale
        + components["poa_ground_diffuse"] * ground_scale
    )


def _view_scale(actual_view: float, unobstructed_view: float) -> float:
    if unobstructed_view <= 1e-9:
        return 0.0
    return float(np.clip(actual_view / unobstructed_view, 0.0, 1.0))
