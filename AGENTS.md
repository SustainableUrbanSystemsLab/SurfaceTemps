# Agent Guide — SurfaceTemps

Frequency-domain admittance method for outdoor surface temperatures. Sub-second per surface, no mesh, no warm-up. Full pipeline: EPW → sol-air → FFT → transfer matrix → IFFT → surface temperature.

## Run & test

```bash
uv pip install -e .          # install (includes pillow, numpy, pvlib, pandas, matplotlib)
uv run pytest                # 16 tests, ~5 s
uv run python examples/neighborhood.py   # full simulation + 3 figures + GIF
```

## Module map

| File | Responsibility |
|---|---|
| `constants.py` | Material property database, Stefan-Boltzmann, R_si / R_so defaults |
| `materials.py` | `Layer` (transfer matrix), `Assembly` (product of matrices), factory functions |
| `weather.py` | `load_epw()` → `WeatherData`; solar position cached as property |
| `solar.py` | `transpose_irradiance()`, `sky_temperature()`, `sol_air_temperature()` |
| `convection.py` | DOE-2 wind model: `h_c = 5.7 + 3.8 * v_wind` |
| `admittance.py` | `solve_surface_temperature()` — core FFT/IFFT solver |
| `geometry.py` | `Box`, `Face`, `GroundPatch`, `NeighborhoodGeometry.create_default()` |
| `surfaces.py` | `build_surfaces()`, `simulate_all()` — orchestrator |
| `plotting.py` | `plot_surface_temps()`, `plot_heatmap()`, `plot_neighborhood_3d()`, `render_daily_gif()` |

## Key non-obvious facts

**Transfer function sign**: `H_n = 1 / (1 + m1 * R_so / m2)`, not `1 + m1/m2 * R_so`. The paper's Eq. 11 has a sign inconsistency from mixed heat-flux conventions. The correct form damps harmonics (`|H_n| < 1`); the paper's form amplifies them.

**R_so excluded from matrix**: `Assembly.total_matrix()` includes R_si and all layers but NOT R_so. R_so is applied as a boundary condition in the solver. Adding it to the matrix and the boundary condition double-counts it.

**Layer order**: `Assembly.layers` runs inside → outside (R_si first, outermost layer last). The matrix product preserves this order: `M = M_si @ M_layer1 @ ... @ M_layerN`.

**Absorptivity ordering**: Without evapotranspiration, higher absorptivity → warmer surface. Grass (α=0.75) is warmer than concrete (α=0.65) in the model. Integration tests verify this.

**Irradiance caching**: `simulate_all()` caches POA irradiance by `(tilt, azimuth)` — walls of the same orientation share one pvlib call.

**Colorbar ticks in plots**: All plots use `BoundaryNorm` with 12 discrete levels. Colorbar ticks are set at boundary values, not level centres.

## Data

`data/atlanta_tmy3.epw` — Atlanta Hartsfield-Jackson TMY3 file, used as the default EPW. Tests and examples depend on it at this path.

## Adding a surface type

1. Add material properties to `constants.py` → `MATERIALS` dict.
2. Add a factory function in `materials.py` returning an `Assembly`.
3. Add `GroundPatch` or `Box` entries in `geometry.py`.
4. Tests in `test_integration.py` will pick up new surfaces automatically.
