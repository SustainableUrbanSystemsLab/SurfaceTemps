"""Keep the Python/C# cross-validation fixture honest.

The fixture in ``data/crossvalidation/admittance_cases.json`` is what the Eddy3D C# port
asserts against. If Python's behaviour changes and the fixture is not regenerated, the C# side
keeps passing against stale expectations and the two implementations drift apart silently —
so this test regenerates every case and compares.

A failure here means one of two things, and the fix differs:
  * you intentionally changed the solver -> rerun ``uv run python scripts/generate_crossvalidation.py``
    AND copy the regenerated file into the Eddy3D repository;
  * you did not -> you have an unintended numerical regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from surface_temps.admittance import solve_surface_temperature
from surface_temps.materials import Assembly, Layer

FIXTURE = (
    Path(__file__).resolve().parents[1] / "data" / "crossvalidation" / "admittance_cases.json"
)


def _load() -> dict:
    if not FIXTURE.exists():
        pytest.fail(
            f"{FIXTURE} is missing — run: uv run python scripts/generate_crossvalidation.py"
        )
    return json.loads(FIXTURE.read_text())


def test_fixture_is_present_and_well_formed():
    data = _load()
    assert data["schema_version"] == 1
    assert data["dt_seconds"] == 3600.0
    assert len(data["cases"]) >= 10, "fixture should cover a range of assemblies and drivers"
    for case in data["cases"]:
        n = data["n_hours"]
        assert len(case["T_driving"]) == n
        assert len(case["T_surface_expected"]) == n
        assert case["assembly"]["R_so"] > 0


def test_every_case_still_reproduces():
    data = _load()
    for case in data["cases"]:
        spec = case["assembly"]
        assembly = Assembly(
            layers=[Layer(**layer) for layer in spec["layers"]],
            R_si=spec["R_si"],
            R_so=spec["R_so"],
        )
        T_driving = np.asarray(case["T_driving"], dtype=float)
        expected = np.asarray(case["T_surface_expected"], dtype=float)

        actual = solve_surface_temperature(
            T_driving, assembly, T_internal=case["T_internal"]
        )

        assert np.max(np.abs(actual - expected)) < 1e-9, (
            f"case '{case['name']}' no longer reproduces. If the solver changed on purpose, "
            "regenerate the fixture and copy it into the Eddy3D repository."
        )


def test_cases_actually_exercise_the_solver():
    """Guard against a fixture of trivial cases that any implementation would pass.

    A port could get the transfer function badly wrong and still reproduce a constant series,
    so at least some cases must show real damping and a real phase shift.
    """
    data = _load()
    damped = 0
    for case in data["cases"]:
        drive = np.asarray(case["T_driving"], dtype=float)
        surf = np.asarray(case["T_surface_expected"], dtype=float)
        drive_swing = drive.max() - drive.min()
        surf_swing = surf.max() - surf.min()
        if drive_swing > 1.0 and surf_swing < 0.95 * drive_swing:
            damped += 1
    assert damped >= 5, f"only {damped} cases show damping — fixture is too weak to catch a port bug"
