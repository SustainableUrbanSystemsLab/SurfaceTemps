"""The outdoor material library: load, validate, and turn entries into assemblies.

The library is data, not code — ``data/materials/outdoor_materials.json`` is the canonical
source and the Eddy3D C# port embeds a byte-identical copy, so a material added here reaches
both implementations without either being edited.

Each entry describes the EXPOSED OUTER LAYER, since that is what sets the surface temperature
the downstream MRT calculation consumes. Layers behind it are named substrates and matter only
through the thermal mass they lend the outer layer — which for a metal skin is everything,
because the skin itself stores nothing over a diurnal cycle.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from surface_temps.constants import R_SI_GROUND, R_SI_WALL, R_SO
from surface_temps.materials import Assembly, Layer

LIBRARY_PATH = Path(__file__).resolve().parents[1] / "data" / "materials" / "outdoor_materials.json"

DIURNAL_PERIOD_S = 86400.0


class MaterialLibraryError(ValueError):
    """Raised when the library file is malformed or a property is out of physical range."""


@dataclass(frozen=True)
class OutdoorMaterial:
    """One exposed outdoor surface."""

    id: str
    name: str
    category: str
    absorptivity: float
    emissivity: float
    conductivity: float
    density: float
    specific_heat: float
    thickness: float
    substrate: tuple[str, ...]
    boundary: str
    estimated: tuple[str, ...]
    source: str
    notes: str

    @property
    def cyclic_thickness(self) -> float:
        """Paper Eq. 5: tau = L*sqrt(pi*rho*c/(P*lambda)) over a diurnal period.

        Below ~0.01 the layer stores no heat over a day and its temperature is dictated by
        whatever sits behind it.
        """
        return self.thickness * math.sqrt(
            math.pi * self.density * self.specific_heat
            / (DIURNAL_PERIOD_S * self.conductivity)
        )

    @property
    def is_thermally_thin(self) -> bool:
        return self.cyclic_thickness < 0.01


class MaterialLibrary:
    """The loaded library, with validation applied."""

    def __init__(self, payload: dict):
        self._raw = payload
        self.validation = payload["validation"]
        self.substrates = payload["substrates"]
        self.materials: dict[str, OutdoorMaterial] = {}

        for entry in payload["materials"]:
            material = OutdoorMaterial(
                id=entry["id"],
                name=entry["name"],
                category=entry["category"],
                absorptivity=float(entry["absorptivity"]),
                emissivity=float(entry["emissivity"]),
                conductivity=float(entry["conductivity"]),
                density=float(entry["density"]),
                specific_heat=float(entry["specific_heat"]),
                thickness=float(entry["thickness"]),
                substrate=tuple(entry["substrate"]),
                boundary=entry["boundary"],
                estimated=tuple(entry.get("estimated", ())),
                source=entry.get("source", ""),
                notes=entry.get("notes", ""),
            )
            self._validate(material)
            if material.id in self.materials:
                raise MaterialLibraryError(f"duplicate material id '{material.id}'")
            self.materials[material.id] = material

    # -- validation -------------------------------------------------------------------

    def _validate(self, m: OutdoorMaterial) -> None:
        v = self.validation
        checks = [
            ("absorptivity", m.absorptivity, v["absorptivity"]),
            ("emissivity", m.emissivity, v["emissivity"]),
            ("conductivity", m.conductivity, v["conductivity"]),
            ("density", m.density, v["density"]),
            ("specific_heat", m.specific_heat, v["specific_heat"]),
        ]
        for field, value, bounds in checks:
            if not (bounds["min"] <= value <= bounds["max"]):
                raise MaterialLibraryError(
                    f"{m.id}: {field}={value} outside [{bounds['min']}, {bounds['max']}]"
                )

        for name in m.substrate:
            if name not in self.substrates:
                raise MaterialLibraryError(f"{m.id}: unknown substrate '{name}'")

        if m.boundary not in ("ground", "indoor"):
            raise MaterialLibraryError(f"{m.id}: boundary must be 'ground' or 'indoor'")

        # The structural rule that actually bites: a thermally thin skin with nothing behind it
        # collapses onto the internal boundary condition and returns a near-constant temperature
        # that looks plausible and means nothing.
        min_tau = v["structural"]["min_cyclic_thickness"]
        if m.cyclic_thickness < min_tau and not m.substrate:
            raise MaterialLibraryError(
                f"{m.id}: cyclic thickness {m.cyclic_thickness:.2e} is below {min_tau} "
                "(the layer stores no heat over a day) and it has no substrate, so the solve "
                "would be governed entirely by the internal boundary condition"
            )

    def emissivity_warnings(self) -> list[str]:
        """Entries whose emissivity falls in the physically unusual middle band.

        Emissivity is bimodal — bare metals low, everything else high — so a value between the
        two clusters usually means a typo or a confused source rather than partially oxidised
        metal.
        """
        lo, hi = self.validation["emissivity"]["warn_between"]
        return [
            f"{m.id}: emissivity {m.emissivity} sits between the metal and non-metal clusters"
            for m in self.materials.values()
            if lo < m.emissivity < hi
        ]

    # -- construction -----------------------------------------------------------------

    def assembly(self, material_id: str, R_so: float = R_SO) -> Assembly:
        """Build the layered assembly for a material, outer layer plus its substrate stack.

        ``Assembly.layers`` runs INSIDE to OUTSIDE, while the JSON lists substrates from just
        behind the surface inwards, so the stack is reversed here.
        """
        m = self[material_id]
        layers = [
            Layer(
                thickness=s["thickness"],
                conductivity=s["conductivity"],
                density=s["density"],
                specific_heat=s["specific_heat"],
            )
            for s in (self.substrates[name] for name in reversed(m.substrate))
        ]
        layers.append(
            Layer(
                thickness=m.thickness,
                conductivity=m.conductivity,
                density=m.density,
                specific_heat=m.specific_heat,
            )
        )
        R_si = R_SI_GROUND if m.boundary == "ground" else R_SI_WALL
        return Assembly(layers=layers, R_si=R_si, R_so=R_so)

    def __getitem__(self, material_id: str) -> OutdoorMaterial:
        try:
            return self.materials[material_id]
        except KeyError as exc:
            options = ", ".join(sorted(self.materials))
            raise MaterialLibraryError(
                f"unknown material '{material_id}'. Available: {options}"
            ) from exc

    def __len__(self) -> int:
        return len(self.materials)

    def __iter__(self):
        return iter(self.materials.values())

    def by_category(self, category: str) -> list[OutdoorMaterial]:
        return [m for m in self.materials.values() if m.category == category]


@lru_cache(maxsize=1)
def load_library(path: str | Path | None = None) -> MaterialLibrary:
    """Load and validate the material library (cached)."""
    p = Path(path) if path is not None else LIBRARY_PATH
    if not p.exists():
        raise MaterialLibraryError(f"material library not found at {p}")
    return MaterialLibrary(json.loads(p.read_text()))
