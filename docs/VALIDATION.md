# Validation and paper audit

A line-by-line re-derivation of [Beckett, Owens & Acred (SimBuild 2026)][paper] against this
implementation and against the Eddy3D C# port, plus the validation suite that now guards it.

[paper]: https://publications.ibpsa.org/conference/paper/?id=simbuild2026_1312

## Summary

The core admittance chain is **correct**, and two places where this code departs from the
printed paper are departures *towards* correctness, not away from it. Three real defects were
found and fixed — one of them in the driving signal, worth up to 6 K at peak; one in the
convection treatment, worth up to 9 K at summer daily peaks; and one in the port, which was
painting sunlight onto north-facing walls at night.

The single most useful outcome is not any individual fix but that the solver is now checked
against an **independent time-domain solver**. Before this work, injecting the paper's own
Eq. 4 variant (an 11% amplitude error) passed every existing test.

## What the paper gets wrong, and why this code is right to differ

### Eq. 11 — mixed heat-flux sign conventions

The paper's Eq. 9 takes `q̃ₒ` positive *outward* (it follows from the matrix orientation in
Eq. 6/7), while Eq. 10 takes it positive *inward*. Combining them as printed gives

```
T̃ₛₒ = T̃ₒ (1 + (m₁/m₂) R_so)
```

which **amplifies** the driving signal. Correcting the convention gives `1 − (m₁/m₂)R_so`, and
because Eq. 7 places `R_so` *inside* the matrix (`m₂ = m₁′R_so + m₂′`, primes excluding `R_so`),
that is algebraically identical to what this code does:

```
H = 1 / (1 + m₁′ R_so / m₂′)      with R_so excluded from the matrix
```

So the two notes in `AGENTS.md` — "exclude R_so from the matrix" and "use `1/(1 + m₁R_so/m₂)`" —
are *jointly* the corrected paper, not a deviation from it.

Checked against the closed-form limit: with no material at all the surface must sit at
`T_o · R_si/(R_si + R_so)`. This code reproduces it to 1e-12; the paper's printed form gives a
value greater than `T_o`, which is impossible for a passive assembly. Pinned by
`test_no_slab_is_a_pure_resistance_divider` and `test_transfer_function_damps_at_every_frequency`.

### Eq. 4/5 — a spurious √2 in the characteristic admittance

With `ξ = sqrt(2πλρc/P)` as defined in Eq. 5, the physical admittance is `λp = (1+j)ξ/√2`, but
Eq. 4 prints `z₃ = ξ(1+j)sinh(τ+jτ)` — larger by exactly √2. Because `z₂` carries the reciprocal
factor, `det(M) = 1` still holds, so the error is invisible to the obvious sanity check.

Both codebases use the physically correct `λp` form (ISO 13786). Pinned explicitly by
`test_layer_matrix_uses_physical_admittance_not_the_papers_xi`, which asserts the ratio between
the two forms really is √2 so the test cannot pass vacuously.

## Defects found and fixed

### 1. Sky temperature inflated by an emissivity divisor — up to 6 K at peak

`sky_temperature()` computed `T_sky = (IR/(0.90·σ))^0.25`. The EPW *Horizontal Infrared
Radiation Intensity* field **is** the downwelling long-wave flux, so the effective sky
temperature is `(IR/σ)^0.25` with no emissivity term — the sky's emissivity is already inside
the measured flux. The surface emissivity belongs in `h_r`, not here.

On Atlanta TMY3 the divisor inflated `T_sky` by **+7.5 K** every hour, leaving the sky only
2.9 K below air where the real clear-sky depression is ~10 K — and making the sky *warmer than
the air* for about a third of the year, which an IR-derived sky temperature cannot be.

End to end on horizontal concrete: annual mean **−1.5 K**, peak **68.1 → 61.8 °C**. The old
code systematically under-predicted night-time radiative cooling, which is the very effect the
paper's abstract says is usually overlooked. The C# port had the identical defect.

### 2. `h_r` hardcoded at 5.0 regardless of emissivity

`surfaces.py` fixed `h_r = 5.0` for every surface while each `Surface` already carried its own
emissivity — collected and never used. 5.0 is right for ε ≈ 0.90, so masonry is unaffected
(−0.03 K annual mean), but `h_r = 4εσT³` falls to 1.4 W/m²K at ε = 0.25 and 0.3 at ε = 0.05.

This only became load-bearing with the new material library, which contains metals: modelling
pre-weathered zinc as if it radiated like masonry made it up to **28 K too cool at peak**. Both
implementations now derive `h_r` from the surface's own emissivity.

### 3. Fixed `R_so` against an hourly `h_e` — up to 9 K at summer daily peaks

The sol-air temperature divides absorbed solar by the **hourly** `h_e = h_c + h_r`, while the
solver coupled the surface to the environment through a **fixed** `1/R_so`. With the DOE-2 wind
model, `h_e` spans 10.7–71.1 W/m²K on Atlanta against `R_so = 0.04` (which implies 25), so on a
calm sunny afternoon the absorbed solar was injected roughly **2.3× too strongly**.

Note the trap here: `R_so = 0.04` is very close to `1/mean(h_e) = 0.042`, so the mean looks fine
and the error hides in the tails — exactly where comfort metrics read.

The paper anticipates this in Eq. 20–22 and it was simply unimplemented. Writing
`h_e(t) = h̄ + Δh(t)`, the surface balance gives `T_sol = T_e + (αQ + q_co)/h̄` with
`q_co = Δh·(T_e − T_s)`, solved by iteration.

Validated against a finite-difference solver driven by the **physical** hourly boundary
condition, so the reference never assumes a constant resistance at all:

| | RMSE vs reference | worst hour | summer daily-peak worst |
|---|---|---|---|
| fixed `R_so` | 1.68 K | 14.8 K | 8.9 K |
| Eq. 20–22 | **1.34 K** | **7.5 K** | **1.6 K** |

**One caveat worth knowing:** the undamped iteration *diverges*. Its gain is `|Δh|/h̄`, which
exceeds 1 for a realistic wind series, so any surface closely following its driving temperature
(a thin or insulated outer layer) runs away — an insulated roof reached 3×10⁴ °C before
under-relaxation was added. `solve_surface_temperature_variable_h` damps adaptively and raises
rather than returning nonsense if it still cannot converge.

### 4. Port: sunlight painted on north walls at night

`Weather` clamps sub-horizon hours to elevation 0 **and azimuth 0**. The transposition then
computes `cosAoi = sin(tilt)·cos(0 − azimuth_surface)`, which is exactly **1.0** for a
north-facing wall. Any EPW hour reporting non-zero DNI while the computed elevation is still ≤ 0
— the sunrise/sunset stamp-convention hours — was applied to north facades at full intensity.
The transposition now zeroes the beam whenever the sun is below the horizon.

### 5. Port: odd-length series lost a harmonic

The C# solver zeroed the imaginary part of bin `n/2` unconditionally. For **even** `n` that bin
is the true Nyquist bin and is real for real input, so it was a no-op matching numpy. For **odd**
`n` there is no Nyquist bin and `n/2` is an ordinary harmonic, so a genuine phase shift was
discarded. Latent for the 8760-hour path; now guarded on parity.

## What was checked and found correct

- **The admittance solver core.** Agreement with an independent Crank-Nicolson finite-difference
  solver converges at second order — exactly 4.0× per grid refinement, down to **5×10⁻⁶ K**.
- **Python ↔ C# numerical parity**, now pinned to 1e-9 by a shared fixture of 10 cases.
- The pyviewfactor matrix orientation, the sun-vector/pvlib azimuth convention, and the absence
  of double-counting between ground-reflected solar and the modelled ground view factor.
- The existing material property values all fall inside published ranges; the problem was
  coverage, not accuracy.

## Known limitations (not defects)

- **No evapotranspiration.** Grass and green roofs are therefore over-predicted during the day;
  measurements put natural turf near 30 °C where artificial turf reaches 50 °C under the same
  weather, and this model cannot reproduce that gap.
- **Surroundings pinned at air temperature.** The non-sky part of the radiant environment uses
  air temperature rather than the surface temperatures the model itself computes. Next to
  sunlit pavement (which runs ~7 K above air on average, p95 +15 K) this cool-biases walls by
  roughly 0.7 K during the day. The paper explicitly anticipates this term being added.
- **Isotropic sky transposition** understates sun-facing walls against Perez by ~8–12 W/m²
  day-mean. The paper prescribes no transposition model.
- **Water** is conduction-only here; evaporation and mixing dominate a real water body.

## The validation suite

| File | What it establishes |
|---|---|
| `tests/reference_fd.py` | Independent time-domain solver; shares no code with `surface_temps` |
| `tests/test_fea_validation.py` | The paper's Figure 3 comparison, plus second-order convergence |
| `tests/test_analytic_limits.py` | Closed-form limits: resistance divider, semi-infinite solid, thermally thin, det(M)=1 |
| `tests/test_variable_convection.py` | Eq. 20–22 beats a fixed `R_so`, and stays bounded |
| `tests/test_crossvalidation_fixture.py` | Keeps the Python↔C# fixture current |
| `tests/test_material_library.py` | Library coverage, physical ordering, and the traps it must reject |
| `tests/test_driving_signal.py` | Regression guards for the three fixed defects |
| `tests/test_shared_data_sync.py` | Digests of the two files shared with the Eddy3D port |

Reproduce the whole thing with `uv run pytest` (64 tests, ~23 s).

## Proving the suite can tell right from wrong

A green suite means nothing by itself — the previous one passed while the solver carried an 11%
amplitude error. `scripts/mutation_check.py` injects each error mode we care about and asserts
the suite catches it:

```
caught  paper-eq4-xi-admittance                by test_layer_matrix_uses_physical_admittance_not_the_papers_xi
caught  paper-eq11-sign-flip                   by test_sinusoidal_amplitude_reduced
caught  conjugated-transfer-function           by test_every_case_still_reproduces
caught  sky-temp-emissivity-divisor-returns    by test_sky_temperature_is_the_blackbody_equivalent
caught  hr-hardcoded-again                     by test_h_radiative_scales_with_emissivity
caught  layer-order-reversed                   by test_every_case_still_reproduces
caught  rso-double-counted-in-matrix           by test_no_slab_is_a_pure_resistance_divider
caught  variable-convection-correction-dropped by test_qco_correction_carries_the_improvement
caught  material-library-emissivity-ignored    by test_properties_are_physically_ordered

All 9 mutations were caught. The suite discriminates.
```

This is worth re-running after any change to the solver or the driving signal. On its first
run it found three genuine holes: nothing pinned the sky-temperature physics, nothing pinned
`h_r` against emissivity on the Python side, and the variable-convection test compared against
the *old* code path rather than isolating the corrective flux — so zeroing `q_co` left the
suite green. `tests/test_driving_signal.py` exists to close exactly those.
