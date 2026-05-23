from pathlib import Path

import numpy as np
import pytest

from surface_temps.geometry import NeighborhoodGeometry
from surface_temps.surfaces import build_surfaces, simulate_all
from surface_temps.weather import load_epw

EPW_PATH = Path(__file__).parent.parent / "data" / "atlanta_tmy3.epw"


@pytest.fixture(scope="module")
def simulation():
    weather = load_epw(EPW_PATH)
    geometry = NeighborhoodGeometry.create_default()
    surfaces = build_surfaces(geometry, weather)
    results = simulate_all(surfaces, weather)
    return results, weather, surfaces


class TestIntegration:
    def test_all_surfaces_have_8760_values(self, simulation):
        results, _, surfaces = simulation
        for surf in surfaces:
            assert surf.name in results
            assert len(results[surf.name]) == 8760

    def test_temperatures_bounded(self, simulation):
        results, _, _ = simulation
        for name, T in results.items():
            assert np.all(T > -50), f"{name} has temps below -50°C"
            # Dark insulated roofs can exceed 100°C sol-air in hot climates
            assert np.all(T < 120), f"{name} has temps above 120°C"

    def test_higher_absorptivity_gives_higher_temps(self, simulation):
        """Without evapotranspiration, higher absorptivity = warmer surface."""
        results, weather, _ = simulation
        summer = slice(4344, 6552)
        if "street_0" in results and "grass_0" in results:
            concrete_mean = np.mean(results["street_0"][summer])  # alpha=0.65
            grass_mean = np.mean(results["grass_0"][summer])  # alpha=0.75
            assert grass_mean > concrete_mean

    def test_south_wall_warmer_than_north_wall_annual(self, simulation):
        results, _, _ = simulation
        south_walls = [k for k in results if "wall_180" in k]
        north_walls = [k for k in results if "wall_000" in k]
        if south_walls and north_walls:
            south_mean = np.mean([np.mean(results[k]) for k in south_walls])
            north_mean = np.mean([np.mean(results[k]) for k in north_walls])
            assert south_mean > north_mean

    def test_surfaces_warmer_than_air_during_summer_day(self, simulation):
        """Horizontal surfaces should be warmer than air during peak summer hours."""
        results, weather, _ = simulation
        # Peak summer afternoon (July, 2pm-ish)
        peak_hours = range(4500, 4510)
        for name in ["street_0", "grass_0"]:
            if name in results:
                surface_mean = np.mean([results[name][h] for h in peak_hours])
                air_mean = np.mean([weather.temp_air[h] for h in peak_hours])
                assert surface_mean > air_mean, f"{name} should be warmer than air"
