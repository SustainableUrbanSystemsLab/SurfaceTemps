from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from surface_temps.geometry import (
    DEFAULT_BUILDINGS_STL,
    DEFAULT_GROUND_STL,
    DEFAULT_MAPPING_JSON,
    NeighborhoodGeometry,
)

TARGET_PANEL_SIZE_M = 5.0


def main() -> None:
    geometry = NeighborhoodGeometry.create_procedural_default()
    data_dir = DEFAULT_BUILDINGS_STL.parent
    data_dir.mkdir(parents=True, exist_ok=True)

    building_triangles, building_mappings = _building_triangles(geometry)
    ground_triangles, ground_mappings = _ground_triangles(geometry)

    _write_ascii_stl(DEFAULT_BUILDINGS_STL, "surface_temps_buildings", building_triangles)
    _write_ascii_stl(DEFAULT_GROUND_STL, "surface_temps_ground", ground_triangles)

    mapping = {
        "version": 1,
        "target_panel_size_m": TARGET_PANEL_SIZE_M,
        "files": {
            "buildings": DEFAULT_BUILDINGS_STL.name,
            "ground": DEFAULT_GROUND_STL.name,
        },
        "surfaces": building_mappings + ground_mappings,
    }
    DEFAULT_MAPPING_JSON.write_text(json.dumps(mapping, indent=2) + "\n")


def _building_triangles(geometry: NeighborhoodGeometry) -> tuple[list[np.ndarray], list[dict]]:
    triangles: list[np.ndarray] = []
    mappings: list[dict] = []

    for box in geometry.boxes:
        for face in box.faces():
            surface_type = "roof" if face.tilt == 0 else "wall"
            assembly = "concrete_roof" if surface_type == "roof" else "brick_wall"
            absorptivity = box.roof_absorptivity if surface_type == "roof" else box.wall_absorptivity
            emissivity = box.roof_emissivity if surface_type == "roof" else box.wall_emissivity

            for local_index, triangle in enumerate(_triangulate(face.vertices)):
                cell = len(triangles)
                triangles.append(triangle)
                mappings.append(
                    {
                        "mesh": "buildings",
                        "cell": cell,
                        "name": f"{face.name}_tri_{local_index}",
                        "view_group": face.name,
                        "surface_type": surface_type,
                        "assembly": assembly,
                        "absorptivity": absorptivity,
                        "emissivity": emissivity,
                        "boundary": "indoor",
                    }
                )

    return triangles, mappings


def _ground_triangles(geometry: NeighborhoodGeometry) -> tuple[list[np.ndarray], list[dict]]:
    triangles: list[np.ndarray] = []
    mappings: list[dict] = []
    assembly_by_material = {
        "concrete": "concrete_ground",
        "brick": "brick_ground",
        "grass": "grass_ground",
    }

    for patch in geometry.ground_patches:
        for local_index, triangle in enumerate(_triangulate(patch.vertices)):
            cell = len(triangles)
            triangles.append(triangle)
            mappings.append(
                {
                    "mesh": "ground",
                    "cell": cell,
                    "name": f"{patch.name}_tri_{local_index}",
                    "view_group": patch.name,
                    "surface_type": "ground",
                    "assembly": assembly_by_material[patch.material_type],
                    "absorptivity": patch.absorptivity,
                    "emissivity": patch.emissivity,
                    "boundary": "ground",
                }
            )

    return triangles, mappings


def _triangulate(vertices: np.ndarray) -> list[np.ndarray]:
    if len(vertices) == 4:
        return _triangulate_refined_quad(vertices, TARGET_PANEL_SIZE_M)
    return [
        np.array([vertices[0], vertices[i], vertices[i + 1]], dtype=float)
        for i in range(1, len(vertices) - 1)
    ]


def _triangulate_refined_quad(vertices: np.ndarray, target_panel_size: float) -> list[np.ndarray]:
    u_divisions = _division_count(vertices[1] - vertices[0], target_panel_size)
    v_divisions = _division_count(vertices[3] - vertices[0], target_panel_size)

    triangles: list[np.ndarray] = []
    for i in range(u_divisions):
        u0 = i / u_divisions
        u1 = (i + 1) / u_divisions
        for j in range(v_divisions):
            v0 = j / v_divisions
            v1 = (j + 1) / v_divisions
            p00 = _quad_point(vertices, u0, v0)
            p10 = _quad_point(vertices, u1, v0)
            p11 = _quad_point(vertices, u1, v1)
            p01 = _quad_point(vertices, u0, v1)
            triangles.append(np.array([p00, p10, p11], dtype=float))
            triangles.append(np.array([p00, p11, p01], dtype=float))
    return triangles


def _division_count(edge_vector: np.ndarray, target_panel_size: float) -> int:
    length = float(np.linalg.norm(edge_vector))
    return max(1, int(np.ceil(length / target_panel_size)))


def _quad_point(vertices: np.ndarray, u: float, v: float) -> np.ndarray:
    return (
        (1.0 - u) * (1.0 - v) * vertices[0]
        + u * (1.0 - v) * vertices[1]
        + u * v * vertices[2]
        + (1.0 - u) * v * vertices[3]
    )


def _write_ascii_stl(path: Path, solid_name: str, triangles: list[np.ndarray]) -> None:
    lines = [f"solid {solid_name}"]
    for triangle in triangles:
        normal = _normal(triangle)
        lines.append(
            f"  facet normal {normal[0]:.12g} {normal[1]:.12g} {normal[2]:.12g}"
        )
        lines.append("    outer loop")
        for vertex in triangle:
            lines.append(
                f"      vertex {vertex[0]:.12g} {vertex[1]:.12g} {vertex[2]:.12g}"
            )
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {solid_name}")
    path.write_text("\n".join(lines) + "\n")


def _normal(triangle: np.ndarray) -> np.ndarray:
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return normal / norm


if __name__ == "__main__":
    main()
