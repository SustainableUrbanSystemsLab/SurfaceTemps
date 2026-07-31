from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numba import njit

from surface_temps.solar import sun_direction_vectors
from surface_temps.weather import WeatherData


@dataclass
class RadiationContext:
    mesh: Any
    triangles: np.ndarray
    triangle_mins: np.ndarray
    triangle_maxs: np.ndarray
    sky_view: np.ndarray
    ground_view: np.ndarray
    building_view: np.ndarray
    ray_length: float
    surface_to_group: np.ndarray
    group_centroids: np.ndarray
    group_normals: np.ndarray
    sunlit_cache: dict[int, np.ndarray] = field(default_factory=dict)


def build_radiation_context(
    surfaces: list[Any],
    *,
    compute_view_factors: bool = True,
    rounding_decimal: int = 5,
) -> RadiationContext | None:
    """Build obstruction geometry and optional pyViewFactor view factors."""
    indexed = [
        (i, surface)
        for i, surface in enumerate(surfaces)
        if getattr(surface, "face_vertices", None) is not None
    ]
    if not indexed:
        return None

    mesh = _polydata_from_surfaces([surface for _, surface in indexed])
    triangles = _triangles_from_mesh(mesh)
    bounds = np.array(mesh.bounds, dtype=float)
    extent = bounds[[1, 3, 5]] - bounds[[0, 2, 4]]
    ray_length = max(float(np.linalg.norm(extent)) * 2.0, 100.0)
    triangle_mins = np.min(triangles, axis=1)
    triangle_maxs = np.max(triangles, axis=1)
    n_total = len(surfaces)
    group_data = _view_factor_groups(indexed, n_total)
    view_mesh = _polydata_from_vertices(group_data["polygons"])

    sky_view = np.ones(n_total)
    ground_view = np.zeros(n_total)
    building_view = np.zeros(n_total)

    if compute_view_factors:
        import pyviewfactor as pvf

        vf = pvf.compute_viewfactor_matrix(
            view_mesh,
            obstacles=view_mesh,
            strict_visibility=False,
            strict_obstruction=False,
            rounding_decimal=rounding_decimal,
            epsilon=1e-4,
            verbose=False,
        )
        vf = np.nan_to_num(vf, nan=0.0, posinf=0.0, neginf=0.0)
        vf = np.clip(vf, 0.0, 1.0)

        ground_targets = np.array(
            [surface_type == "ground" for surface_type in group_data["surface_types"]]
        )
        building_targets = ~ground_targets

        for source_col, surface_indices in enumerate(group_data["indices"]):
            to_scene = vf[:, source_col]
            ground = float(np.sum(to_scene[ground_targets]))
            building = float(np.sum(to_scene[building_targets]))
            scene = ground + building
            if scene > 1.0:
                ground /= scene
                building /= scene
                scene = 1.0

            for surface_index in surface_indices:
                ground_view[surface_index] = ground
                building_view[surface_index] = building
                sky_view[surface_index] = max(0.0, 1.0 - scene)
    else:
        for surface_index, surface in indexed:
            tilt = getattr(surface, "tilt", 0.0)
            sky_view[surface_index] = ideal_sky_view_factor(tilt)
            ground_view[surface_index] = ideal_ground_view_factor(tilt)

    return RadiationContext(
        mesh=mesh,
        triangles=triangles,
        triangle_mins=triangle_mins,
        triangle_maxs=triangle_maxs,
        sky_view=sky_view,
        ground_view=ground_view,
        building_view=building_view,
        ray_length=ray_length,
        surface_to_group=group_data["surface_to_group"],
        group_centroids=group_data["centroids"],
        group_normals=group_data["normals"],
    )


def sunlit_factors(
    surface_index: int,
    surface: Any,
    weather: WeatherData,
    context: RadiationContext | None,
    *,
    eps: float = 1e-4,
) -> np.ndarray:
    """Return 1 for direct-sun-visible hours and 0 for obstructed hours."""
    n_hours = len(weather.dni)
    if context is None or getattr(surface, "face_vertices", None) is None:
        return np.ones(n_hours)
    group_index = int(context.surface_to_group[surface_index])
    cache_key = group_index if group_index >= 0 else surface_index
    if cache_key in context.sunlit_cache:
        return context.sunlit_cache[cache_key]

    if group_index >= 0:
        normal = context.group_normals[group_index]
        centroid = context.group_centroids[group_index]
    else:
        vertices = np.asarray(surface.face_vertices, dtype=float)
        normal = _face_normal(vertices)
        centroid = np.mean(vertices, axis=0)
    directions = sun_direction_vectors(weather)
    facing = directions @ normal
    active = (weather.dni > 1.0) & (directions[:, 2] > 0.0) & (facing > 1e-6)

    start = centroid + normal * eps
    factors = _sunlit_factors_jit(
        start.astype(np.float64),
        directions.astype(np.float64),
        active.astype(np.bool_),
        context.triangles.astype(np.float64),
        context.triangle_mins.astype(np.float64),
        context.triangle_maxs.astype(np.float64),
        float(context.ray_length),
        float(eps),
    )
    context.sunlit_cache[cache_key] = factors
    return factors


def ideal_sky_view_factor(tilt: float) -> float:
    return float((1.0 + np.cos(np.radians(tilt))) / 2.0)


def ideal_ground_view_factor(tilt: float) -> float:
    return float((1.0 - np.cos(np.radians(tilt))) / 2.0)


def _polydata_from_surfaces(surfaces: list[Any]):
    import pyvista as pv

    points: list[list[float]] = []
    faces: list[int] = []
    for surface in surfaces:
        vertices = np.asarray(surface.face_vertices, dtype=float)
        start = len(points)
        points.extend(vertices.tolist())
        faces.extend([len(vertices), *range(start, start + len(vertices))])
    return pv.PolyData(np.asarray(points, dtype=float), np.asarray(faces))


def _view_factor_groups(indexed_surfaces: list[tuple[int, Any]], n_surfaces: int):
    groups: dict[str, list[tuple[int, Any]]] = {}
    for surface_index, surface in indexed_surfaces:
        group_key = getattr(surface, "view_group", "") or getattr(surface, "name", "")
        groups.setdefault(group_key, []).append((surface_index, surface))

    surface_to_group = np.full(n_surfaces, -1, dtype=int)
    group_indices: list[list[int]] = []
    group_surface_types: list[str] = []
    group_polygons: list[np.ndarray] = []
    group_centroids: list[np.ndarray] = []
    group_normals: list[np.ndarray] = []

    for group_index, group in enumerate(groups.values()):
        surface_indices = [surface_index for surface_index, _ in group]
        surfaces = [surface for _, surface in group]
        for surface_index in surface_indices:
            surface_to_group[surface_index] = group_index
        group_indices.append(surface_indices)
        group_surface_types.append(getattr(surfaces[0], "surface_type", ""))
        polygon = _convex_planar_hull(surfaces)
        group_polygons.append(polygon)
        group_centroids.append(np.mean(polygon, axis=0))
        group_normals.append(_face_normal(polygon))

    return {
        "indices": group_indices,
        "surface_types": group_surface_types,
        "polygons": group_polygons,
        "surface_to_group": surface_to_group,
        "centroids": np.asarray(group_centroids, dtype=float),
        "normals": np.asarray(group_normals, dtype=float),
    }


def _polydata_from_vertices(polygons: list[np.ndarray]):
    import pyvista as pv

    points: list[list[float]] = []
    faces: list[int] = []
    for polygon in polygons:
        start = len(points)
        points.extend(polygon.tolist())
        faces.extend([len(polygon), *range(start, start + len(polygon))])
    return pv.PolyData(np.asarray(points, dtype=float), np.asarray(faces))


def _convex_planar_hull(surfaces: list[Any]) -> np.ndarray:
    vertices = np.vstack([np.asarray(surface.face_vertices, dtype=float) for surface in surfaces])
    vertices = np.unique(np.round(vertices, decimals=8), axis=0)

    normal = _face_normal(np.asarray(surfaces[0].face_vertices, dtype=float))
    origin = np.mean(vertices, axis=0)
    axis_u = vertices[0] - origin
    for candidate in vertices[1:]:
        axis_u = candidate - origin
        if np.linalg.norm(axis_u) > 1e-9:
            break
    axis_u = axis_u / np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    axis_v = axis_v / np.linalg.norm(axis_v)

    projected = np.column_stack(
        [
            (vertices - origin) @ axis_u,
            (vertices - origin) @ axis_v,
        ]
    )
    hull_indices = _convex_hull_indices(projected)
    polygon = _remove_collinear_vertices(vertices[hull_indices])
    if np.dot(_face_normal(polygon), normal) < 0:
        polygon = polygon[::-1]
    return polygon


def _convex_hull_indices(points_2d: np.ndarray) -> list[int]:
    order = sorted(range(len(points_2d)), key=lambda idx: (points_2d[idx, 0], points_2d[idx, 1]))

    def cross(o, a, b):
        return (points_2d[a, 0] - points_2d[o, 0]) * (points_2d[b, 1] - points_2d[o, 1]) - (
            points_2d[a, 1] - points_2d[o, 1]
        ) * (points_2d[b, 0] - points_2d[o, 0])

    lower: list[int] = []
    for idx in order:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], idx) <= 0:
            lower.pop()
        lower.append(idx)

    upper: list[int] = []
    for idx in reversed(order):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], idx) <= 0:
            upper.pop()
        upper.append(idx)

    return lower[:-1] + upper[:-1]


def _remove_collinear_vertices(vertices: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    keep = list(vertices)
    changed = True
    while changed and len(keep) > 3:
        changed = False
        filtered = []
        n = len(keep)
        for i, current in enumerate(keep):
            prev = keep[(i - 1) % n]
            nxt = keep[(i + 1) % n]
            if np.linalg.norm(np.cross(current - prev, nxt - current)) <= tol:
                changed = True
                continue
            filtered.append(current)
        keep = filtered
    return np.asarray(keep, dtype=float)


def _triangles_from_mesh(mesh: Any) -> np.ndarray:
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    return np.asarray([mesh.points[cell] for cell in faces], dtype=float)


def _face_normal(vertices: np.ndarray) -> np.ndarray:
    normal = np.zeros(3)
    for i, current in enumerate(vertices):
        nxt = vertices[(i + 1) % len(vertices)]
        normal[0] += (current[1] - nxt[1]) * (current[2] + nxt[2])
        normal[1] += (current[2] - nxt[2]) * (current[0] + nxt[0])
        normal[2] += (current[0] - nxt[0]) * (current[1] + nxt[1])
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return normal / norm


@njit(cache=True)
def _sunlit_factors_jit(
    start,
    directions,
    active,
    triangles,
    triangle_mins,
    triangle_maxs,
    ray_length,
    eps,
):
    factors = np.zeros(active.shape[0], dtype=np.float64)
    for hour in range(active.shape[0]):
        if not active[hour]:
            continue
        end = start + directions[hour] * ray_length
        ray_min = np.minimum(start, end) - eps
        ray_max = np.maximum(start, end) + eps
        blocked = False
        for tri_index in range(triangles.shape[0]):
            if not _aabb_overlaps(
                ray_min,
                ray_max,
                triangle_mins[tri_index],
                triangle_maxs[tri_index],
            ):
                continue
            tri = triangles[tri_index]
            if _segment_intersects_triangle(
                start,
                end,
                tri[0],
                tri[1],
                tri[2],
                eps,
            ):
                blocked = True
                break
        if not blocked:
            factors[hour] = 1.0
    return factors


@njit(cache=True)
def _aabb_overlaps(a_min, a_max, b_min, b_max):
    return (
        a_min[0] <= b_max[0]
        and a_max[0] >= b_min[0]
        and a_min[1] <= b_max[1]
        and a_max[1] >= b_min[1]
        and a_min[2] <= b_max[2]
        and a_max[2] >= b_min[2]
    )


@njit(cache=True)
def _segment_intersects_triangle(orig, dest, v0, v1, v2, eps):
    d = dest - orig
    if np.sqrt(np.dot(d, d)) < eps:
        return False

    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(d, edge2)
    a = np.dot(edge1, h)
    if abs(a) < eps:
        return False

    f = 1.0 / a
    s = orig - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False

    q = np.cross(s, edge1)
    v = f * np.dot(d, q)
    if v < 0.0 or (u + v) > 1.0:
        return False

    t = f * np.dot(edge2, q)
    return (t > eps) and (t < 1.0 - eps)
