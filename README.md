# SurfaceTemps

Frequency-domain admittance method for computing outdoor surface temperatures, based on [Beckett, Owens, and Acred (SimBuild 2026)](https://doi.org/10.26868/25746308.2026.1312).

The method decomposes the sol-air driving temperature into Fourier harmonics, applies a material-specific transfer matrix at each frequency, and reconstructs the surface temperature via inverse FFT. No mesh, no warm-up period, sub-second computation per surface.

## Quickstart

```bash
uv pip install -e .
uv run python examples/neighborhood.py
```

This simulates a full year of hourly surface temperatures for a 12-building neighborhood in Atlanta and produces three figures:

- **3D neighborhood snapshot** at the peak summer hour
- **Time-series** of selected surface temperatures over 3 summer days
- **Heatmap** (hour-of-day vs day-of-year) for a concrete street

## Method overview

1. **Material characterisation** — Multi-layer assemblies (brick walls, concrete slabs, soil+grass) are described by a 2x2 complex transfer matrix at each frequency.
2. **Sol-air temperature** — Combines air temperature, sky/ground radiant temperature, and incident solar radiation into a single driving signal per surface orientation.
3. **FFT** — Decomposes the 8760-hour sol-air signal into harmonics.
4. **Admittance transfer function** — Each harmonic is multiplied by H_n = 1/(1 + m1·R_so/m2), where m1, m2 come from the transfer matrix at that frequency.
5. **IFFT** — Reconstructs the surface temperature time series.

Solar position and irradiance transposition are handled by [pvlib](https://pvlib-python.readthedocs.io/). Weather data comes from standard EPW files.

## Default neighborhood

The example creates 12 box-shaped buildings of varying height (5–15 m), footprint, and rotation (0°–60°), arranged on a ~200 m grid with concrete streets, grass lawns, and brick courtyards.

| Surface type | Material layers | Solar absorptivity |
|---|---|---|
| Concrete street | 1 m subsoil + 200 mm concrete | 0.65 |
| Brick paving | 1 m subsoil + 100 mm sand + 65 mm brick | 0.70 |
| Grass | 1 m subsoil + 300 mm topsoil + 50 mm sod | 0.75 |
| Brick wall | 13 mm plaster + 150 mm block + 25 mm air gap + 102 mm brick | 0.70 |
| Concrete roof | 13 mm plaster + 150 mm concrete + 35 mm insulation + 15 mm screed + 1 mm membrane | 0.85 |

## Project structure

```
surface_temps/
    constants.py        Material properties, physical constants
    materials.py        Layer and Assembly transfer matrix computation
    weather.py          EPW loading via pvlib
    solar.py            Sol-air temperature, sky temperature, irradiance transposition
    convection.py       Wind-dependent convective heat transfer coefficient
    admittance.py       Core FFT/IFFT solver
    geometry.py         Box buildings, ground patches, neighborhood layout
    surfaces.py         Simulation orchestrator
    plotting.py         2D time-series, heatmaps, 3D neighborhood visualization
```

## Testing

```bash
uv run pytest
```

Tests verify transfer matrix determinant = 1, steady-state and sinusoidal response correctness, temperature bounds, orientation effects (south walls warmer than north), and absorptivity ordering.

## Dependencies

- numpy
- pvlib
- pandas
- matplotlib

## Reference

Beckett, O., Owens, S., and Acred, A. (2026). "Applying Frequency Domain Methods for Calculating Outdoor Surface Temperatures." *Proceedings of the 12th National Conference of IBPSA-USA*, Minneapolis, MN.
