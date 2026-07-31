"""Simulate surface temperatures for a 12-box neighborhood and visualize in 3D."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from surface_temps.geometry import NeighborhoodGeometry
from surface_temps.plotting import (
    plot_heatmap,
    plot_neighborhood_3d,
    plot_surface_temps,
    render_daily_gif,
)
from surface_temps.surfaces import build_surfaces, simulate_all
from surface_temps.weather import load_epw

EPW_PATH = Path(__file__).parent.parent / "data" / "atlanta_tmy3.epw"


def main():
    print("Loading weather data...")
    weather = load_epw(EPW_PATH)
    print(f"  Location: {weather.latitude:.2f}°N, {weather.longitude:.2f}°E")
    print(f"  Annual mean air temp: {weather.annual_mean_temp:.1f} °C")

    print("\nCreating neighborhood geometry...")
    geometry = NeighborhoodGeometry.create_default()
    if geometry.mesh_surfaces:
        building_faces = [s for s in geometry.mesh_surfaces if s.mesh_name == "buildings"]
        ground_faces = [s for s in geometry.mesh_surfaces if s.mesh_name == "ground"]
        print(
            f"  {len(building_faces)} building facets, "
            f"{len(ground_faces)} ground facets from STL"
        )
    else:
        print(f"  {len(geometry.boxes)} buildings, {len(geometry.ground_patches)} ground patches")

    print("\nBuilding surfaces...")
    surfaces = build_surfaces(geometry, weather)
    print(f"  {len(surfaces)} total surfaces")

    print("\nSimulating (full year, hourly)...")
    results = simulate_all(surfaces, weather)
    print("  Done.")

    # Summary statistics
    print("\n--- Summary ---")
    for name, T in sorted(results.items()):
        print(f"  {name:25s}  mean={np.mean(T):6.1f}°C  max={np.max(T):6.1f}°C  min={np.min(T):6.1f}°C")

    # Find peak summer hour (hottest air temperature hour)
    peak_hour = int(np.argmax(weather.temp_air))
    print(f"\nPeak air temp hour: {peak_hour} ({weather.temp_air[peak_hour]:.1f}°C)")

    # 3D snapshot at peak hour
    fig1 = plot_neighborhood_3d(geometry, results, hour=peak_hour)
    fig1.savefig("neighborhood_3d_peak.png", dpi=150, bbox_inches="tight")
    print("Saved: neighborhood_3d_peak.png")

    # Time series: 3 summer days starting from peak
    start = max(0, peak_hour - 24)
    sample_prefixes = ["street_0", "grass_0", "brick_0", "B01_wall_180", "B01_roof"]
    available = []
    for prefix in sample_prefixes:
        match = next((name for name in results if name.startswith(prefix)), None)
        if match is not None:
            available.append(match)
    fig2 = plot_surface_temps(results, weather, start_hour=start, num_hours=72, surface_names=available)
    fig2.savefig("surface_temps_timeseries.png", dpi=150, bbox_inches="tight")
    print("Saved: surface_temps_timeseries.png")

    # Heatmap for a concrete street
    street_surface = next((name for name in results if name.startswith("street_0")), None)
    if street_surface is not None:
        fig3 = plot_heatmap(results, street_surface)
        fig3.savefig("heatmap_street.png", dpi=150, bbox_inches="tight")
        print("Saved: heatmap_street.png")

    # Animated GIF: 24h cycle on the peak day
    peak_day_start = (peak_hour // 24) * 24
    print(f"\nRendering 24-hour GIF (day starting at hour {peak_day_start})...")
    render_daily_gif(geometry, results, day_start_hour=peak_day_start, output_path="neighborhood_24h.gif")
    print("Saved: neighborhood_24h.gif")

    plt.show()


if __name__ == "__main__":
    main()
