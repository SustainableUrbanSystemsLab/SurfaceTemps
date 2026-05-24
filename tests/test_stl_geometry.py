from __future__ import annotations

import numpy as np
import pandas as pd

from surface_temps.geometry import NeighborhoodGeometry
from surface_temps.materials import brick_wall, concrete_ground
from surface_temps.radiation import build_radiation_context
from surface_temps.surfaces import Surface, simulate_all
from surface_temps.weather import WeatherData


def test_default_geometry_loads_from_stl_mapping():
    geometry = NeighborhoodGeometry.create_default()

    assert not geometry.boxes
    assert not geometry.ground_patches
    assert len(geometry.mesh_surfaces) == 3178

    surface_types = [surface.surface_type for surface in geometry.mesh_surfaces]
    assert surface_types.count("wall") == 712
    assert surface_types.count("roof") == 244
    assert surface_types.count("ground") == 2222
    assert geometry.mesh_surfaces[0].name == "B01_wall_180_tri_0"


def test_view_factor_context_reduces_sky_view_behind_wall():
    shaded, open_ground, wall = _shadow_test_surfaces()
    context = build_radiation_context([shaded, open_ground, wall])

    assert context is not None
    assert context.sky_view[0] < context.sky_view[1]
    assert context.building_view[0] > context.building_view[1]


def test_direct_occlusion_lowers_surface_temperature():
    weather = _fixed_south_sun_weather(24)
    shaded, open_ground, wall = _shadow_test_surfaces()

    results = simulate_all(
        [shaded, open_ground, wall],
        weather,
        use_occlusion=True,
        use_view_factors=False,
    )

    assert np.max(results["shaded_ground"]) < np.max(results["open_ground"])


def _shadow_test_surfaces() -> tuple[Surface, Surface, Surface]:
    shaded = Surface(
        name="shaded_ground",
        assembly=concrete_ground(),
        tilt=0,
        azimuth=0,
        absorptivity=0.65,
        emissivity=0.90,
        T_internal=15.0,
        face_vertices=np.array([[-1.0, 0.5, 0.0], [1.0, 0.5, 0.0], [0.0, 2.5, 0.0]]),
        surface_type="ground",
    )
    open_ground = Surface(
        name="open_ground",
        assembly=concrete_ground(),
        tilt=0,
        azimuth=0,
        absorptivity=0.65,
        emissivity=0.90,
        T_internal=15.0,
        face_vertices=np.array([[8.0, 0.5, 0.0], [10.0, 0.5, 0.0], [9.0, 2.5, 0.0]]),
        surface_type="ground",
    )
    wall = Surface(
        name="shade_wall",
        assembly=brick_wall(),
        tilt=90,
        azimuth=0,
        absorptivity=0.70,
        emissivity=0.90,
        T_internal=22.0,
        face_vertices=np.array(
            [
                [-2.0, -0.2, 0.0],
                [-2.0, -0.2, 5.0],
                [2.0, -0.2, 5.0],
                [2.0, -0.2, 0.0],
            ]
        ),
        surface_type="wall",
    )
    return shaded, open_ground, wall


def _fixed_south_sun_weather(n_hours: int) -> WeatherData:
    values = np.full(n_hours, 25.0)
    times = pd.date_range("2026-06-21", periods=n_hours, freq="h", tz="Etc/GMT+5")
    weather = WeatherData(
        temp_air=values.copy(),
        dew_point=np.full(n_hours, 15.0),
        ghi=np.full(n_hours, 900.0),
        dni=np.full(n_hours, 800.0),
        dhi=np.full(n_hours, 100.0),
        wind_speed=np.full(n_hours, 1.0),
        infrared_horizontal=np.full(n_hours, 350.0),
        latitude=33.64,
        longitude=-84.43,
        altitude=313.0,
        tz=-5.0,
        location=None,
        times=times,
    )
    weather._solar_position = pd.DataFrame(
        {
            "apparent_zenith": np.full(n_hours, 30.0),
            "azimuth": np.full(n_hours, 180.0),
        },
        index=times,
    )
    return weather
