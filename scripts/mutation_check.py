"""Prove the test suite can tell right from wrong, by breaking the physics on purpose.

A passing suite says nothing on its own. Before this work, the suite here passed even when the
solver was given the paper's Eq. 4 variant — an 11% amplitude error — because nothing pinned a
time-varying value against an independent reference.

This injects each error mode we actually care about and reports whether the suite CATCHES it.
Every mutation must produce at least one failing test. A mutation that survives is a hole.

Run: ``uv run python scripts/mutation_check.py``
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


@dataclass
class Mutation:
    name: str
    path: Path
    old: str
    new: str
    why: str


MUTATIONS = [
    Mutation(
        name="paper-eq4-xi-admittance",
        path=REPO / "surface_temps" / "materials.py",
        old="        z2 = np.sinh(pL) / (lam * p)\n        z3 = lam * p * np.sinh(pL)",
        new=(
            "        _xi = np.sqrt(2 * np.pi * lam * self.density * self.specific_heat / period)\n"
            "        z2 = np.sinh(pL) / (_xi * (1 + 1j))\n"
            "        z3 = _xi * (1 + 1j) * np.sinh(pL)"
        ),
        why="the paper's printed Eq. 4: sqrt(2) too large, det(M)=1 still holds so it hides",
    ),
    Mutation(
        name="paper-eq11-sign-flip",
        path=REPO / "surface_temps" / "admittance.py",
        old="        H[n] = 1.0 / (1.0 + m1 * R_so / m2)",
        new="        H[n] = 1.0 + m1 * R_so / m2",
        why="the paper's literal Eq. 11, which amplifies instead of damping",
    ),
    Mutation(
        name="conjugated-transfer-function",
        path=REPO / "surface_temps" / "admittance.py",
        old="        H[n] = 1.0 / (1.0 + m1 * R_so / m2)",
        new="        H[n] = np.conj(1.0 / (1.0 + m1 * R_so / m2))",
        why="right amplitude, inverted phase: the surface would lead rather than lag the sun",
    ),
    Mutation(
        name="sky-temp-emissivity-divisor-returns",
        path=REPO / "surface_temps" / "solar.py",
        old="    T_sky_K[valid] = (ir[valid] / STEFAN_BOLTZMANN) ** 0.25",
        new="    T_sky_K[valid] = (ir[valid] / (0.90 * STEFAN_BOLTZMANN)) ** 0.25",
        why="the fixed defect coming back: +7.5 K sky, +1.5 K on every surface",
    ),
    Mutation(
        name="hr-hardcoded-again",
        path=REPO / "surface_temps" / "solar.py",
        old="    return 4.0 * emissivity * STEFAN_BOLTZMANN * T_K**3",
        new="    return np.full_like(np.asarray(T_K, dtype=float), 5.0)",
        why="ignoring emissivity again: metals become 28 K too cool at peak",
    ),
    Mutation(
        name="layer-order-reversed",
        path=REPO / "surface_temps" / "materials.py",
        old="        for layer in self.layers:\n            M = M @ layer.transfer_matrix(period)",
        new="        for layer in reversed(self.layers):\n            M = M @ layer.transfer_matrix(period)",
        why="inside/outside flipped: the sun would see the wrong material",
    ),
    Mutation(
        name="rso-double-counted-in-matrix",
        path=REPO / "surface_temps" / "materials.py",
        old="        M_si = np.array([[1, self.R_si], [0, 1]], dtype=complex)",
        new=(
            "        M_si = np.array([[1, self.R_si], [0, 1]], dtype=complex) @ "
            "np.array([[1, self.R_so], [0, 1]], dtype=complex)"
        ),
        why="R_so in the matrix AND as a boundary condition, the double-count AGENTS.md warns of",
    ),
    Mutation(
        name="variable-convection-correction-dropped",
        path=REPO / "surface_temps" / "admittance.py",
        old="    scale = delta_h / h_bar",
        new="    scale = np.zeros_like(delta_h)",
        why="paper Eq. 20-22 silently disabled: calm sunny peaks over-predicted again",
    ),
    Mutation(
        name="material-library-emissivity-ignored",
        path=REPO / "surface_temps" / "library.py",
        old='                emissivity=float(entry["emissivity"]),',
        new="                emissivity=0.90,",
        why="every material forced to masonry emissivity, erasing the metals' distinction",
    ),
]


def run_suite() -> tuple[bool, str]:
    proc = subprocess.run(
        ["uv", "run", "pytest", "-x", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def first_failing_test(output: str) -> str:
    hits = re.findall(r"FAILED (\S+)", output)
    if hits:
        return hits[0]
    hits = re.findall(r"^(tests/\S+?)::(\S+)", output, re.M)
    return f"{hits[0][0]}::{hits[0][1]}" if hits else "(unattributed failure)"


def main() -> int:
    print("Baseline: the suite must be green before any mutation means anything.")
    ok, output = run_suite()
    if not ok:
        print("BASELINE IS RED — fix that first.\n")
        print(output[-3000:])
        return 2
    print("  baseline green\n")

    survivors = []
    for m in MUTATIONS:
        backup = m.path.with_suffix(m.path.suffix + ".mutbak")
        shutil.copy(m.path, backup)
        try:
            source = m.path.read_text()
            if m.old not in source:
                print(
                    f"  SKIP  {m.name}: anchor text not found in {m.path.name} "
                    "(the code moved; update this mutation)"
                )
                survivors.append((m, "anchor-missing"))
                continue
            m.path.write_text(source.replace(m.old, m.new, 1))

            caught, output = run_suite()
            if caught:  # suite still green => mutation survived
                print(f"  SURVIVED  {m.name}  <-- {m.why}")
                survivors.append((m, "not-detected"))
            else:
                print(f"  caught    {m.name}  by {first_failing_test(output)}")
        finally:
            shutil.copy(backup, m.path)
            backup.unlink()

    print()
    if survivors:
        print(f"{len(survivors)} of {len(MUTATIONS)} mutations SURVIVED — the suite has holes:")
        for m, reason in survivors:
            print(f"  - {m.name} ({reason}): {m.why}")
        return 1

    print(f"All {len(MUTATIONS)} mutations were caught. The suite discriminates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
