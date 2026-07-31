"""Guard the two shared data files against one-sided edits.

``data/materials/outdoor_materials.json`` and ``data/crossvalidation/admittance_cases.json``
are the canonical copies; the Eddy3D C# port embeds byte-identical duplicates and asserts the
SAME digests in ``Radiance.Core.Tests/TestSharedDataSync.cs``.

Because both repositories pin the same constants, editing one copy and forgetting the other
turns exactly one of them red — which is the only mechanism that makes "shared" data actually
shared across two repositories that cannot see each other at test time.

When you legitimately change a shared file:
  1. regenerate it (for the fixture: ``uv run python scripts/generate_crossvalidation.py``);
  2. update the digest here;
  3. copy the file into ``Eddy3D/Radiance.Core/Resources/`` and update the digest there too.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Keep in lockstep with Radiance.Core.Tests/TestSharedDataSync.cs.
EXPECTED = {
    REPO / "data" / "materials" / "outdoor_materials.json":
        "0b669044bdd0a9ffb7ec8bc9db40b59ce7b66a28043bfd64bf6f52b2e0ff5adb",
    REPO / "data" / "crossvalidation" / "admittance_cases.json":
        "75bd85ac0b55bf079a87f3cec0f902981bc4c38ce65388f7ca08242836ce61fe",
}


@pytest.mark.parametrize("path", list(EXPECTED), ids=lambda p: p.name)
def test_shared_file_digest_is_pinned(path: Path):
    assert path.exists(), f"{path} is missing"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == EXPECTED[path], (
        f"{path.name} changed.\n"
        f"  expected {EXPECTED[path]}\n"
        f"  actual   {digest}\n"
        "If the change is intended: update this digest, copy the file into "
        "Eddy3D/Radiance.Core/Resources/, and update the matching digest in "
        "Radiance.Core.Tests/TestSharedDataSync.cs. Otherwise the two implementations have "
        "silently diverged."
    )
