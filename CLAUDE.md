# Agent Guide — SurfaceTemps

Frequency-domain admittance method for outdoor surface temperatures. STL geometry, per-surface admittance solve, no thermal mesh, no warm-up. Full pipeline: EPW → STL/JSON surfaces → occlusion/view factors → sol-air → FFT → transfer matrix → IFFT → surface temperature.

## Run & test

```bash
uv pip install -e .          # install (includes pyviewfactor, pyvista, numba, pillow, numpy, pvlib, pandas, matplotlib)
uv run pytest                # 52 tests, ~25 s; default integration skips full-scene occlusion/view factors
uv run python examples/neighborhood.py   # full simulation + 3 figures + GIF
```

## Module map

| File | Responsibility |
|---|---|
| `constants.py` | Legacy material property database, Stefan-Boltzmann, R_si / R_so defaults |
| `library.py` | The outdoor material library: loads/validates `data/materials/outdoor_materials.json`, builds assemblies |
| `materials.py` | `Layer` (transfer matrix), `Assembly` (product of matrices), factory functions |
| `weather.py` | `load_epw()` → `WeatherData`; solar position cached as property |
| `solar.py` | `transpose_irradiance()`, component transposition, sun vectors, `sky_temperature()`, `sol_air_temperature()` |
| `radiation.py` | pyViewFactor view factors, direct sun obstruction, sky/ground/building view factors |
| `convection.py` | DOE-2 wind model: `h_c = 5.7 + 3.8 * v_wind` |
| `admittance.py` | `solve_surface_temperature()` — core FFT/IFFT solver; `solve_surface_temperature_variable_h()` — paper Eq. 20-22 |
| `geometry.py` | `Box`, `Face`, `GroundPatch`, `MeshSurface`, STL/JSON import, procedural default source |
| `surfaces.py` | `build_surfaces()`, `simulate_all()` — orchestrator |
| `plotting.py` | `plot_surface_temps()`, `plot_heatmap()`, `plot_neighborhood_3d()`, `render_daily_gif()` |

## Key non-obvious facts

**Transfer function sign**: `H_n = 1 / (1 + m1 * R_so / m2)`, not `1 + m1/m2 * R_so`. The paper's Eq. 11 has a sign inconsistency from mixed heat-flux conventions. The correct form damps harmonics (`|H_n| < 1`); the paper's form amplifies them.

**R_so excluded from matrix**: `Assembly.total_matrix()` includes R_si and all layers but NOT R_so. R_so is applied as a boundary condition in the solver. Adding it to the matrix and the boundary condition double-counts it.

**Layer order**: `Assembly.layers` runs inside → outside (R_si first, outermost layer last). The matrix product preserves this order: `M = M_si @ M_layer1 @ ... @ M_layerN`.

**Absorptivity ordering**: Without evapotranspiration, higher absorptivity → warmer surface. Grass (α=0.75) is warmer than concrete (α=0.65) in the model. Integration tests verify this.

**Sky temperature has NO emissivity divisor**: `T_sky = (IR/σ)^0.25`. The EPW horizontal-IR field IS the downwelling long-wave flux, so the sky's emissivity is already in it. Dividing by 0.90 (as this used to) inflates T_sky by 7.5 K, leaves the sky warmer than the air a third of the year, and biases every surface ~1.5 K warm. Surface emissivity belongs in `h_r`.

**h_r is derived per surface from ε**, not fixed at 5.0: `h_r = 4εσT³`. 5.0 is right only near ε=0.90. The library contains metals at ε=0.25 and below, where the true h_r is ~1.4 and a hardcoded 5.0 makes them up to 28 K too cool at peak.

**Variable convection (paper Eq. 20-22) is ON by default.** Sol-air divides absorbed solar by hourly `h_e`, but the solver couples through a fixed `1/R_so`; on a calm sunny afternoon that over-injects solar 2.3×. Beware: `R_so=0.04 ≈ 1/mean(h_e)`, so the error hides in the tails, not the mean. **The iteration diverges undamped** (gain `|Δh|/h̄` > 1) — an insulated roof reached 3e4 °C before under-relaxation; `solve_surface_temperature_variable_h` damps adaptively.

**The material library is shared data, not code**: `data/materials/outdoor_materials.json` is canonical and the Eddy3D C# port embeds a byte-identical copy. Entries describe the EXPOSED OUTER LAYER (that is what MRT consumes) plus a substrate stack. A thermally thin outer layer (any metal skin, τ<0.01) MUST carry a substrate or the solve collapses onto the internal boundary.

**Validation lives in `docs/VALIDATION.md`** — the paper audit (including two errors in the paper itself: the Eq. 11 sign and a √2 in Eq. 4/5) and the reasoning behind every tolerance. `tests/reference_fd.py` is an independent time-domain solver; agreement converges at second order to 5e-6 K. Before it existed, injecting the paper's own Eq. 4 variant (11% amplitude error) passed every test.

**Default geometry source**: `NeighborhoodGeometry.create_default()` reads `data/neighborhood_buildings.stl`, `data/neighborhood_ground.stl`, and `data/neighborhood_surfaces.json`. `create_procedural_default()` remains as the source used by `scripts/generate_default_stl.py`.

**JSON mapping**: STL has no material metadata. `data/neighborhood_surfaces.json` maps `(mesh, cell)` to name, surface type, assembly, absorptivity, emissivity, and boundary.

**Occlusion**: Direct beam solar is blocked by STL triangle ray intersections. Diffuse solar and longwave radiant temperature use pyViewFactor visibility/obstruction view factors. Sky view is the complement of modeled building and ground view factors.

**5 m mesh acceleration**: `data/neighborhood_surfaces.json` includes `view_group` parent IDs. `radiation.py` computes direct-shadow masks and pyViewFactor view factors on those parent rectangles, then applies them to the 5 m child triangles. This avoids a dense 3178 x 3178 pyViewFactor matrix and redundant annual ray tests.

**Irradiance caching**: `simulate_all()` caches POA irradiance components by `(tilt, azimuth)` — walls of the same orientation share one pvlib call before per-facet occlusion/view-factor adjustment.

**Colorbar ticks in plots**: All plots use `BoundaryNorm` with 12 discrete levels. Colorbar ticks are set at boundary values, not level centres.

## Data

- `data/atlanta_tmy3.epw` — Atlanta Hartsfield-Jackson TMY3 file, used as the default EPW. Tests and examples depend on it at this path.
- `data/neighborhood_buildings.stl` — default 12-building STL, 956 triangular facets.
- `data/neighborhood_ground.stl` — default streets/grass/brick ground STL, 2,222 triangular facets.
- `data/neighborhood_surfaces.json` — material and boundary mapping for both STL files.

## Adding a surface type

1. Add material properties to `constants.py` → `MATERIALS` dict.
2. Add a factory function in `materials.py` returning an `Assembly`.
3. Add or update STL facets and entries in `data/neighborhood_surfaces.json`.
4. For the default procedural source, update `geometry.py` and rerun `uv run python scripts/generate_default_stl.py`.
5. Tests in `test_integration.py` and `test_stl_geometry.py` will pick up new surfaces automatically.
