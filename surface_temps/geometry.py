from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from surface_temps.materials import (
    Assembly,
    assembly_from_name,
    brick_ground,
    brick_wall,
    concrete_ground,
    concrete_roof,
    grass_ground,
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_BUILDINGS_STL = DEFAULT_DATA_DIR / "neighborhood_buildings.stl"
DEFAULT_GROUND_STL = DEFAULT_DATA_DIR / "neighborhood_ground.stl"
DEFAULT_MAPPING_JSON = DEFAULT_DATA_DIR / "neighborhood_surfaces.json"


@dataclass
class Face:
    vertices: np.ndarray  # (4, 3) corner positions
    tilt: float  # 0 = horizontal (roof/ground), 90 = vertical (wall)
    azimuth: float  # degrees from north, clockwise
    area: float  # m2
    name: str = ""


@dataclass
class Box:
    center_x: float  # m
    center_y: float  # m
    width: float  # m (x-extent before rotation)
    length: float  # m (y-extent before rotation)
    height: float  # m
    rotation: float = 0.0  # degrees from north, clockwise
    wall_assembly: Assembly = field(default_factory=brick_wall)
    roof_assembly: Assembly = field(default_factory=concrete_roof)
    wall_absorptivity: float = 0.70
    roof_absorptivity: float = 0.85
    wall_emissivity: float = 0.90
    roof_emissivity: float = 0.90
    label: str = ""

    def _rotate_point(self, x: float, y: float) -> tuple[float, float]:
        theta = np.radians(self.rotation)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rx = x * cos_t - y * sin_t
        ry = x * sin_t + y * cos_t
        return rx + self.center_x, ry + self.center_y

    def _base_corners(self) -> list[tuple[float, float]]:
        hw, hl = self.width / 2, self.length / 2
        local = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
        return [self._rotate_point(x, y) for x, y in local]

    def faces(self) -> list[Face]:
        corners = self._base_corners()
        h = self.height
        result = []

        bottom = np.array([(x, y, 0) for x, y in corners])
        top = np.array([(x, y, h) for x, y in corners])

        wall_azimuths = [180, 270, 0, 90]  # S, W, N, E for unrotated box
        for i in range(4):
            j = (i + 1) % 4
            verts = np.array([bottom[i], bottom[j], top[j], top[i]])
            az = (wall_azimuths[i] + self.rotation) % 360

            dx = corners[j][0] - corners[i][0]
            dy = corners[j][1] - corners[i][1]
            wall_width = np.sqrt(dx**2 + dy**2)
            area = wall_width * h

            name = f"{self.label}_wall_{int(az):03d}" if self.label else f"wall_{int(az):03d}"
            result.append(Face(verts, tilt=90, azimuth=az, area=area, name=name))

        roof_verts = top.copy()
        roof_area = self.width * self.length
        name = f"{self.label}_roof" if self.label else "roof"
        result.append(Face(roof_verts, tilt=0, azimuth=0, area=roof_area, name=name))

        return result


@dataclass
class GroundPatch:
    vertices: np.ndarray  # (N, 3) polygon vertices at z=0
    assembly: Assembly = field(default_factory=concrete_ground)
    absorptivity: float = 0.65
    emissivity: float = 0.90
    material_type: str = "concrete"
    name: str = ""


@dataclass
class MeshSurface:
    vertices: np.ndarray
    tilt: float
    azimuth: float
    area: float
    name: str
    assembly: Assembly
    absorptivity: float
    emissivity: float
    surface_type: str
    boundary: str
    mesh_name: str
    cell_index: int
    view_group: str = ""


@dataclass
class NeighborhoodGeometry:
    boxes: list[Box] = field(default_factory=list)
    ground_patches: list[GroundPatch] = field(default_factory=list)
    mesh_surfaces: list[MeshSurface] = field(default_factory=list)

    @classmethod
    def create_default(cls) -> NeighborhoodGeometry:
        if (
            DEFAULT_BUILDINGS_STL.exists()
            and DEFAULT_GROUND_STL.exists()
            and DEFAULT_MAPPING_JSON.exists()
        ):
            return cls.from_stl(
                building_stl=DEFAULT_BUILDINGS_STL,
                ground_stl=DEFAULT_GROUND_STL,
                mapping_json=DEFAULT_MAPPING_JSON,
            )
        return cls.create_procedural_default()

    @classmethod
    def create_procedural_default(cls) -> NeighborhoodGeometry:
        box_specs = [
            # (cx, cy, w, l, h, rot, label)
            (30, 30, 15, 25, 12, 0, "B01"),
            (80, 25, 10, 20, 8, 0, "B02"),
            (130, 35, 12, 12, 15, 0, "B03"),
            (30, 90, 20, 10, 6, 30, "B04"),
            (85, 85, 8, 16, 10, 0, "B05"),
            (140, 95, 14, 18, 9, 45, "B06"),
            (35, 145, 10, 10, 7, 0, "B07"),
            (90, 150, 18, 12, 11, 15, "B08"),
            (145, 155, 8, 8, 5, 0, "B09"),
            (30, 200, 16, 22, 14, 60, "B10"),
            (90, 210, 12, 15, 8, 0, "B11"),
            (150, 205, 10, 14, 10, 30, "B12"),
        ]

        boxes = []
        for cx, cy, w, length, h, rot, label in box_specs:
            boxes.append(
                Box(
                    center_x=cx,
                    center_y=cy,
                    width=w,
                    length=length,
                    height=h,
                    rotation=rot,
                    label=label,
                )
            )

        patches = _create_default_ground_patches()
        return cls(boxes=boxes, ground_patches=patches)

    @classmethod
    def from_mapping(cls, mapping_json: str | Path) -> NeighborhoodGeometry:
        mapping_path = Path(mapping_json)
        mapping = _load_mapping(mapping_path)
        files = mapping.get("files", {})
        base_dir = mapping_path.parent

        building_stl = _resolve_mapping_path(base_dir, files.get("buildings"))
        ground_stl = _resolve_mapping_path(base_dir, files.get("ground"))
        if building_stl is None:
            raise ValueError(f"Mapping file {mapping_path} is missing files.buildings")

        return cls.from_stl(
            building_stl=building_stl,
            ground_stl=ground_stl,
            mapping_json=mapping_path,
        )

    @classmethod
    def from_stl(
        cls,
        building_stl: str | Path,
        ground_stl: str | Path | None = None,
        mapping_json: str | Path | None = None,
    ) -> NeighborhoodGeometry:
        """Load building and ground facets from STL files plus a JSON sidecar."""
        import pyvista as pv

        mapping = _load_mapping(mapping_json) if mapping_json is not None else {}
        by_mesh = _mapping_by_mesh_and_cell(mapping)
        mesh_specs: list[tuple[str, Path]] = [("buildings", Path(building_stl))]
        if ground_stl is not None:
            mesh_specs.append(("ground", Path(ground_stl)))

        surfaces: list[MeshSurface] = []
        for mesh_name, stl_path in mesh_specs:
            mesh = pv.read(stl_path).extract_surface(algorithm="dataset_surface")
            if not mesh.is_all_triangles:
                mesh = mesh.triangulate()

            for cell_index in range(mesh.n_cells):
                vertices = mesh.get_cell(cell_index).points.astype(float)
                normal = _face_normal(vertices)
                tilt, azimuth = _tilt_azimuth_from_normal(normal)
                area = _polygon_area(vertices)
                attrs = by_mesh.get(mesh_name, {}).get(cell_index)
                if attrs is None:
                    attrs = _default_surface_mapping(mesh_name, cell_index, normal)

                surfaces.append(
                    MeshSurface(
                        vertices=vertices,
                        tilt=tilt,
                        azimuth=azimuth,
                        area=area,
                        name=attrs["name"],
                        assembly=assembly_from_name(attrs["assembly"]),
                        absorptivity=float(attrs["absorptivity"]),
                        emissivity=float(attrs["emissivity"]),
                        surface_type=attrs["surface_type"],
                        boundary=attrs["boundary"],
                        mesh_name=mesh_name,
                        cell_index=cell_index,
                        view_group=attrs.get("view_group", attrs["name"]),
                    )
                )

        return cls(mesh_surfaces=surfaces)

    def iter_render_faces(self):
        for surface in self.mesh_surfaces:
            yield surface.name, surface.vertices
        for box in self.boxes:
            for face in box.faces():
                yield face.name, face.vertices
        for patch in self.ground_patches:
            yield patch.name, patch.vertices


def _create_default_ground_patches() -> list[GroundPatch]:
    patches = []
    extent = 190.0

    # Streets: horizontal and vertical corridors
    street_w = 8.0
    street_rects = [
        # Horizontal streets
        (0, 60, extent, 60 + street_w),
        (0, 120, extent, 120 + street_w),
        (0, 175, extent, 175 + street_w),
        # Vertical streets
        (55, 0, 55 + street_w, 240),
        (115, 0, 115 + street_w, 240),
    ]
    for i, (x0, y0, x1, y1) in enumerate(street_rects):
        verts = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])
        patches.append(
            GroundPatch(
                vertices=verts,
                assembly=concrete_ground(),
                absorptivity=0.65,
                material_type="concrete",
                name=f"street_{i}",
            )
        )

    # Grass patches in between buildings
    grass_rects = [
        (0, 0, 20, 20),
        (45, 45, 55, 60),
        (0, 128, 20, 175),
        (63, 128, 115, 175),
        (123, 128, 165, 175),
        (160, 0, 190, 60),
        (160, 128, 190, 175),
    ]
    for i, (x0, y0, x1, y1) in enumerate(grass_rects):
        verts = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])
        patches.append(
            GroundPatch(
                vertices=verts,
                assembly=grass_ground(),
                absorptivity=0.75,
                emissivity=0.95,
                material_type="grass",
                name=f"grass_{i}",
            )
        )

    # Brick courtyards near some buildings
    brick_rects = [
        (63, 0, 115, 25),
        (0, 68, 55, 120),
        (63, 68, 115, 120),
    ]
    for i, (x0, y0, x1, y1) in enumerate(brick_rects):
        verts = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])
        patches.append(
            GroundPatch(
                vertices=verts,
                assembly=brick_ground(),
                absorptivity=0.70,
                material_type="brick",
                name=f"brick_{i}",
            )
        )

    return patches


def _load_mapping(mapping_json: str | Path | None) -> dict:
    if mapping_json is None:
        return {}
    return json.loads(Path(mapping_json).read_text())


def _resolve_mapping_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _mapping_by_mesh_and_cell(mapping: dict) -> dict[str, dict[int, dict]]:
    result: dict[str, dict[int, dict]] = {}
    for item in mapping.get("surfaces", []):
        mesh_name = item["mesh"]
        cell_index = int(item["cell"])
        result.setdefault(mesh_name, {})[cell_index] = item
    return result


def _default_surface_mapping(mesh_name: str, cell_index: int, normal: np.ndarray) -> dict:
    if mesh_name == "ground":
        return {
            "name": f"ground_{cell_index:03d}",
            "view_group": f"ground_{cell_index:03d}",
            "surface_type": "ground",
            "assembly": "concrete_ground",
            "absorptivity": 0.65,
            "emissivity": 0.90,
            "boundary": "ground",
        }
    if normal[2] > 0.75:
        return {
            "name": f"roof_{cell_index:03d}",
            "view_group": f"roof_{cell_index:03d}",
            "surface_type": "roof",
            "assembly": "concrete_roof",
            "absorptivity": 0.85,
            "emissivity": 0.90,
            "boundary": "indoor",
        }
    return {
        "name": f"wall_{cell_index:03d}",
        "view_group": f"wall_{cell_index:03d}",
        "surface_type": "wall",
        "assembly": "brick_wall",
        "absorptivity": 0.70,
        "emissivity": 0.90,
        "boundary": "indoor",
    }


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


def _tilt_azimuth_from_normal(normal: np.ndarray) -> tuple[float, float]:
    z = float(np.clip(normal[2], -1.0, 1.0))
    tilt = float(np.degrees(np.arccos(z)))
    if abs(normal[0]) < 1e-12 and abs(normal[1]) < 1e-12:
        azimuth = 0.0
    else:
        azimuth = float(np.degrees(np.arctan2(normal[0], normal[1])) % 360.0)
    return tilt, azimuth


def _polygon_area(vertices: np.ndarray) -> float:
    normal = _face_normal(vertices)
    area = 0.0
    for i, current in enumerate(vertices):
        nxt = vertices[(i + 1) % len(vertices)]
        area += np.dot(normal, np.cross(current, nxt))
    return float(0.5 * abs(area))
