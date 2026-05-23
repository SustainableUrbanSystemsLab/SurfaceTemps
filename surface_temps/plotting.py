from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm

from surface_temps.geometry import NeighborhoodGeometry
from surface_temps.weather import WeatherData

N_LEVELS = 12


def _discrete_cmap(cmap_name: str, vmin: float, vmax: float, n: int = N_LEVELS):
    """Return a discretized colormap, norm, and boundary array."""
    bounds = np.linspace(vmin, vmax, n + 1)
    cmap = plt.get_cmap(cmap_name, n)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds


def plot_surface_temps(
    results: dict[str, np.ndarray],
    weather: WeatherData,
    start_hour: int = 0,
    num_hours: int = 72,
    surface_names: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Time-series plot of selected surface temperatures vs air temperature."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    else:
        fig = ax.get_figure()

    hours = np.arange(start_hour, start_hour + num_hours)
    sl = slice(start_hour, start_hour + num_hours)

    ax.plot(hours, weather.temp_air[sl], "k--", alpha=0.5, label="Air temperature")

    if surface_names is None:
        surface_names = list(results.keys())

    for name in surface_names:
        if name in results:
            ax.plot(hours, results[name][sl], label=name)

    ax.set_xlabel("Hour of year")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_heatmap(
    results: dict[str, np.ndarray],
    surface_name: str,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Hour-of-day vs day-of-year heatmap of surface temperature."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))
    else:
        fig = ax.get_figure()

    data = results[surface_name][:8760]
    grid = data.reshape(365, 24).T

    vmin = np.floor(grid.min() / 5) * 5
    vmax = np.ceil(grid.max() / 5) * 5
    cmap, norm, bounds = _discrete_cmap("RdYlBu_r", vmin, vmax)

    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        norm=norm,
        extent=[0, 365, 0, 24],
    )
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Hour of day")
    ax.set_title(surface_name)
    cb = plt.colorbar(im, ax=ax, label="Temperature (°C)", spacing="proportional")
    cb.set_ticks(bounds)
    fig.tight_layout()
    return fig


def plot_neighborhood_3d(
    geometry: NeighborhoodGeometry,
    results: dict[str, np.ndarray],
    hour: int,
    ax: plt.Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    elev: float = 35,
    azim: float = -60,
    cmap_name: str = "RdYlBu_r",
) -> plt.Figure:
    """3D visualization of the neighborhood colored by surface temperature."""
    if ax is None:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    all_temps = []
    face_data = []

    for box in geometry.boxes:
        for face in box.faces():
            T = results.get(face.name)
            if T is not None:
                all_temps.append(T[hour])
                face_data.append((face.vertices, T[hour]))

    for patch in geometry.ground_patches:
        T = results.get(patch.name)
        if T is not None:
            all_temps.append(T[hour])
            face_data.append((patch.vertices, T[hour]))

    if not all_temps:
        return fig

    if vmin is None:
        vmin = np.floor(min(all_temps) / 5) * 5
    if vmax is None:
        vmax = np.ceil(max(all_temps) / 5) * 5

    cmap, norm, bounds = _discrete_cmap(cmap_name, vmin, vmax)

    polys = []
    colors = []
    for verts, temp in face_data:
        polys.append(verts[:, :3])
        colors.append(cmap(norm(temp)))

    collection = Poly3DCollection(polys, alpha=0.85, linewidths=0.3)
    collection.set_facecolors(colors)
    collection.set_edgecolors([(0.2, 0.2, 0.2, 0.5)] * len(colors))
    ax.add_collection3d(collection)

    all_verts = np.vstack([v for v, _ in face_data])
    margin = 10
    ax.set_xlim(all_verts[:, 0].min() - margin, all_verts[:, 0].max() + margin)
    ax.set_ylim(all_verts[:, 1].min() - margin, all_verts[:, 1].max() + margin)
    ax.set_zlim(0, all_verts[:, 2].max() + 5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=elev, azim=azim)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(
        sm, ax=ax, label="Surface temperature (°C)", shrink=0.6, pad=0.1
    )
    cb.set_ticks(bounds)

    ax.set_title(f"Neighborhood surface temperatures — hour {hour}")
    return fig
