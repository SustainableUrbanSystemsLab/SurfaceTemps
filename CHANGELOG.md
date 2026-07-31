# CHANGELOG


## v0.0.0 (2026-07-31)

### Code Style

- Adopt ruff format across the codebase
  ([`a2fb617`](https://github.com/SustainableUrbanSystemsLab/SurfaceTemps/commit/a2fb6171a32c3fa9e2776e11cf08a7cbd8d227f0))

The lint job was failing because I left the `ruff format --check` step in ci.yml while claiming in
  the PR description that formatting was not enforced. The two disagreed; this resolves it in favour
  of enforcing it, per request.

`ruff format` reformatted 15 files. The formatter is now the arbiter of layout, so this is the
  one-time cost of adopting it and there should be no further formatting churn.

One hand-edit on top of the formatter: it exploded the shared-data digest table into multi-line
  `REPO / "data" / ... :` dict keys, which is valid but unreadable for a table whose whole job is to
  be scanned by eye. Keyed on repo-relative strings instead, which both reads better and formats
  stably.

Verified after reformatting: 64 tests pass, ruff check and ruff format --check are clean, and the
  mutation check still catches all 9 error modes — worth confirming explicitly, since it anchors on
  exact source text and a reformat could have silently invalidated its anchors.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>

### Continuous Integration

- Add GitHub Actions and automated semantic releases
  ([`ce78395`](https://github.com/SustainableUrbanSystemsLab/SurfaceTemps/commit/ce78395a2aba8807e1146de0c4d3f5a3f7a23cad))

CI (.github/workflows/ci.yml) runs on every push and PR:

- tests on Python 3.12 and 3.13, with a JUnit summary so failures are readable from the checks tab
  rather than the raw log; - the mutation check, which injects nine known error modes and fails if
  any survives. This is the job that actually protects the physics: the suite once passed while the
  solver carried an 11% amplitude error, so "tests are green" is not by itself evidence; - ruff,
  pinned to an explicit ruleset in pyproject rather than ruff's defaults, since defaults change
  between releases and CI that silently gains rules fails for reasons unrelated to the change under
  review.

Releases (.github/workflows/release.yml) are cut by python-semantic-release from Conventional
  Commits, gated on CI going green on main first — a release built from a red commit is worse than
  no release. A manual run defaults to a dry run so the next version can be inspected before it is
  real.

Two version settings are set deliberately, both verified against the real history rather than
  assumed:

- allow_zero_version = true. Without it the first release is forced to 1.0.0 whatever the current
  version says. - major_on_zero = false. Without it the first `feat:` promotes 0.x straight to
  1.0.0.

Together they keep the project in 0.x until someone declares 1.0, which is a statement about API
  stability rather than a side effect of adding a feature. Verified: with a v0.1.0 baseline, a
  `feat:` commit yields 0.2.0.

PyPI publishing is deliberately NOT wired up. It needs either a trusted-publisher configuration or
  an API token, and it pushes artefacts to a public index under this project's name — a maintainer
  decision, not a default to inherit. release.yml documents how to add it.

The lint fixes here are mechanical (ambiguous `l` loop variables, an implicit Optional, an unused
  loop variable, zip -> itertools.pairwise) plus per-file ignores where the rule is wrong for this
  codebase: Greek symbols in docstrings are deliberate physics notation, and fixture tuples keep
  consistent names across tests even where one test does not read every element.

CONTRIBUTING.md documents the commit convention, since versions now depend on it.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
