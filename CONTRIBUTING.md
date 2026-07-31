# Contributing

## Commit messages drive the version

Releases are cut automatically by [python-semantic-release][psr] from
[Conventional Commits][cc], so the commit message is what decides the next version. There is
no manual bump step, and `pyproject.toml`'s version can never disagree with the git tags.

| Prefix | Effect | Example |
|---|---|---|
| `fix:` | patch — `0.1.0` → `0.1.1` | `fix: correct the sky-temperature emissivity divisor` |
| `perf:` | patch | `perf: cache the transfer function per assembly` |
| `feat:` | minor — `0.1.0` → `0.2.0` | `feat: add the outdoor material library` |
| `feat!:` or `BREAKING CHANGE:` in the body | major | `feat!: drop the constant-h solver` |
| `docs:` `test:` `refactor:` `chore:` `ci:` `build:` `style:` | no release | `docs: explain the Eq. 20-22 correction` |

The project deliberately stays in `0.x` until someone declares 1.0 — that is a statement about
API stability, not something a feature commit should trigger.

## Before you push

```bash
uv run pytest                        # 64 tests, ~23 s
uvx ruff check .                     # lint
uv run python scripts/mutation_check.py   # proves the suite still catches known errors
```

The mutation check is the one people skip and shouldn't. A green suite is not evidence on its
own: this one once passed while the solver carried an 11% amplitude error, and the first run of
`mutation_check.py` found three more holes. If you change the solver or the driving signal, run
it — and if you move code it anchors on, update the anchors rather than letting it report a
survivor.

## Physics changes

`docs/VALIDATION.md` records the audit against the source paper, including two errors in the
paper itself and the reasoning behind every tolerance in the suite. If you change a number that
affects results, update it there in the same PR.

[psr]: https://python-semantic-release.readthedocs.io/
[cc]: https://www.conventionalcommits.org/
