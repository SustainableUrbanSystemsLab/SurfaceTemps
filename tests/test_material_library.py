"""The outdoor material library: coverage, physical plausibility, and the traps it must catch."""

from __future__ import annotations

import json

import numpy as np
import pytest

from surface_temps.admittance import solve_surface_temperature
from surface_temps.library import (
    LIBRARY_PATH,
    MaterialLibrary,
    MaterialLibraryError,
    load_library,
)


@pytest.fixture(scope="module")
def lib():
    return load_library()


def test_library_covers_what_a_designer_would_specify(lib):
    assert len(lib) >= 20, "the brief asks for at least 20 commonly specified materials"
    for category in ("paving", "facade", "roof", "vegetation"):
        assert lib.by_category(category), f"no materials in category '{category}'"

    # The urban-heat-island essentials: without these the library cannot express the
    # design decisions it exists to inform.
    for required in (
        "asphalt_new",
        "concrete_paving_grey",
        "natural_grass",
        "bitumen_membrane_black",
        "tpo_white_cool",
        "green_roof_extensive",
        "zinc_preweathered",
        "glass_curtain_wall",
    ):
        assert required in lib.materials, f"missing {required}"


def test_every_entry_is_sourced(lib):
    for m in lib:
        assert m.source, f"{m.id} has no citation"
        for field in m.estimated:
            assert hasattr(m, field), f"{m.id} flags unknown field '{field}' as estimated"


def test_properties_are_physically_ordered(lib):
    """Spot-check the orderings a wrong sign or transposed column would break."""
    # Fresh black asphalt absorbs more than a cool white roof, by a wide margin.
    assert lib["asphalt_new"].absorptivity > lib["tpo_white_cool"].absorptivity + 0.6
    # Metals conduct orders of magnitude better than masonry.
    assert lib["zinc_preweathered"].conductivity > 50 * lib["clay_brick_paver_red"].conductivity
    # Water has by far the highest specific heat.
    assert lib["water_shallow"].specific_heat > 3000
    # Bare metal patina is the low-emissivity outlier; everything non-metallic is high.
    assert lib["zinc_preweathered"].emissivity < 0.4
    for m in lib:
        if m.category in ("paving", "vegetation", "water"):
            assert m.emissivity > 0.8, f"{m.id} emissivity {m.emissivity} too low for a non-metal"


def test_thermally_thin_metals_are_flagged_and_backed(lib):
    """A metal skin stores nothing over a day, so it must carry a substrate.

    Without one the admittance solve degenerates onto the internal boundary condition and
    returns a smooth, plausible, meaningless series.
    """
    thin = [m for m in lib if m.is_thermally_thin]
    assert {m.id for m in thin} >= {"zinc_preweathered", "aluminium_anodised", "corten_rusted"}
    for m in thin:
        assert m.substrate, f"{m.id} is thermally thin (tau={m.cyclic_thickness:.1e}) but unbacked"


def test_loader_rejects_out_of_range_properties():
    payload = json.loads(LIBRARY_PATH.read_text())
    payload["materials"][0]["emissivity"] = 90.0  # percent typed as a fraction
    with pytest.raises(MaterialLibraryError, match="emissivity"):
        MaterialLibrary(payload)


def test_loader_rejects_a_thin_layer_with_no_substrate():
    payload = json.loads(LIBRARY_PATH.read_text())
    zinc = next(m for m in payload["materials"] if m["id"] == "zinc_preweathered")
    zinc["substrate"] = []
    with pytest.raises(MaterialLibraryError, match="cyclic thickness"):
        MaterialLibrary(payload)


def test_loader_rejects_unknown_substrate():
    payload = json.loads(LIBRARY_PATH.read_text())
    payload["materials"][0]["substrate"] = ["not_a_real_substrate"]
    with pytest.raises(MaterialLibraryError, match="unknown substrate"):
        MaterialLibrary(payload)


def test_assemblies_build_and_run(lib):
    """Every entry must produce an assembly that actually solves to a sane temperature."""
    n = 24 * 4
    T_driving = 20.0 + 12.0 * np.sin(2.0 * np.pi * np.arange(n) / 24.0)

    for m in lib:
        assembly = lib.assembly(m.id)
        assert assembly.layers, f"{m.id} produced no layers"
        T = solve_surface_temperature(T_driving, assembly, T_internal=18.0)
        assert np.all(np.isfinite(T)), f"{m.id} produced non-finite temperatures"
        assert -60.0 < T.min() and T.max() < 120.0, f"{m.id} left the plausible range: {T.min()}..{T.max()}"


def test_assembly_layer_order_is_inside_to_outside(lib):
    """The outer layer must be LAST, matching Assembly's contract.

    Reversing this silently swaps which material the sun sees, and the result stays smooth,
    so only an explicit check catches it.
    """
    m = lib["asphalt_new"]
    assembly = lib.assembly("asphalt_new")
    outer = assembly.layers[-1]
    assert outer.conductivity == pytest.approx(m.conductivity)
    assert outer.thickness == pytest.approx(m.thickness)
    # And the deepest substrate (subsoil, 1 m) is first.
    assert assembly.layers[0].thickness == pytest.approx(1.0)


def test_cool_roof_runs_cooler_than_black_roof(lib):
    """The library's headline claim, checked end to end rather than asserted in a comment."""
    n = 24 * 7
    hours = np.arange(n)
    # Crude clear-sky driver: a warm day with a strong solar component.
    T_air = 28.0 + 6.0 * np.sin(2.0 * np.pi * hours / 24.0 - 1.2)
    solar = np.maximum(0.0, 900.0 * np.sin(2.0 * np.pi * (hours % 24) / 24.0 - 1.2))
    h_c, h_r = 12.0, 5.0

    peaks = {}
    for material_id in ("bitumen_membrane_black", "tpo_white_cool"):
        m = lib[material_id]
        T_sol = (h_c * T_air + h_r * (T_air - 8.0) + m.absorptivity * solar) / (h_c + h_r)
        T = solve_surface_temperature(T_sol, lib.assembly(material_id), T_internal=22.0)
        peaks[material_id] = T.max()

    delta = peaks["bitumen_membrane_black"] - peaks["tpo_white_cool"]
    assert delta > 15.0, f"cool roof should be far cooler at peak, got {delta:.1f} K"
