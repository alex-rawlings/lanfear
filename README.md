<p align="center">
<img src="logo/lanfear.png" width="300">
</p>

# LANFEAR: Linear Analytic N-body Field Expansion to Ascertain Resonances

Orbit analysis for galaxy simulations. A C++ core (SCF potentials, orbit
integration, frequency analysis) with a thin Python interface.

The pipeline is:

1. **Read** a snapshot.
2. **Build** an analytical potential from the particles — a Hernquist-Ostriker
   (HO) SCF basis for spherical-ish systems, or a disc basis for flattened
   systems — and **check** it agrees with the simulation potential to `< X%`.
3. **Integrate** the orbits of chosen particle families in that potential for
   `N` orbital periods.
4. **Fourier-transform** each orbit to find resonances along the principal axes.
5. **Classify** orbits (inner/outer x-tube, box, z-tube, rosette, π-box, …).
6. **Store** each particle's classification and orbit properties.

The central black hole is handled specially: an SCF basis cannot represent a
point mass, so the potential is expanded over **everything except the BH**, and
the BH is re-attached afterwards as a softened point mass **at its actual
position** (which need not be the origin).


## Requirements

- Python 3.13, NumPy, SciPy, h5py, pybind11,
  Boost ≥ 1.70, CMake ≥ 3.15).

## Building

```bash
./build.sh
```

This produces `lanfear/_core*.so` in place. Add the repository root to
`PYTHONPATH` (or `cd` here) so `import lanfear` works.

### Installing into a virtual environment

After building the extension with `./build.sh`, install the package with pip so
`import lanfear` works from anywhere in the environment:

```bash
pip install -e .          # editable: picks up the in-place _core*.so
# or
pip install .             # regular install (bundles the built _core*.so)

pip install -e ".[mpi]"   # also install mpi4py for parallel runs
pip install -e ".[dev]"   # ruff, pre-commit, pytest, scipy
pip install -e ".[all]"   # every optional dependency (mpi + build + dev)
```

The compiled extension is ABI-specific to the Python it was built against, so build and install in the same environment; the wheel
is not portable across machines/Python versions.

## Usage

```python
import lanfear as lf

ps = lf.ParticleSystem.from_gadget_hdf5("snapshot.hdf5")
ps.prepare()             # recentre, align, scale radius, figure-rotation check

# Spherical-ish systems: Hernquist-Ostriker basis.
pot = lf.Potential.from_particles(ps, n_max=18, l_max=7)
# Flattened / disc-like systems: Miyamoto-Nagai disc basis (same interface).
#   pot = lf.DiscPotential.from_particles(ps, n_radial=10, n_vert=3)
result = pot.validate()                       # analytic potential vs direct sum
print(result)                                 # median / p90 / worst rel. error
assert result.passed(tolerance=0.02)

# Evaluate (HO units) at physical coordinates:
phi = pot.potential([[1.0, 0.0, 0.0]])
acc = pot.acceleration([[1.0, 0.0, 0.0]])

# Integrate every star for 50 orbital periods (MPI-distributed if launched
# under srun/mpirun, otherwise serial):
res = lf.integrate_family(pot, ps, family="STAR", n_periods=50)
if res is not None:                           # None on non-root MPI ranks
    print(res.column("energy_drift"))         # per-orbit summary columns
    print(lf.SUMMARY_COLUMNS)                 # available quantities
    good = res.ok                             # status == 0

# Or integrate AND frequency-analyse (fundamentals + spectral lines per axis):
res = lf.analyse_family(pot, ps, family="STAR", n_periods=50, n_lines=4)
if res is not None:
    res.fundamentals        # (N, 3) signed fundamental frequency per axis (HO)
    res.lines               # (N, 3, n_lines, 2) leading (freq, amp) per axis
    res.frequency_ratios    # (N, 2) |w_x|/|w_z|, |w_y|/|w_z|

    # Integration is expensive -- save the results and reload later without
    # re-integrating (a compact .npz holding everything OrbitResults needs):
    res.save("orbits.npz")
    res = lf.OrbitResults.load("orbits.npz")   # resume classification/plotting

    # Classify into orbit families (box / tube / rosette / boxlet ...):
    cls = res.classify()
    cls.labels              # (N,) lf.OrbitClass values
    cls.names               # (N,) family name strings
    cls.counts()            # {family_name: count}
    z_tubes = cls.mask(lf.OrbitClass.SHORT_AXIS_TUBE)

    # Condense the subclasses into the box / tube dichotomy:
    fam = cls.condense_families()
    fam.counts()            # {'box': ..., 'tube': ..., 'unclassified': ...}
    tubes = fam.mask(lf.OrbitFamily.TUBE)

    # Radial profile of the orbit-class mix (returns a matplotlib Axes):
    import numpy as np
    ax = cls.plot_class_fractions(np.linspace(0, 20, 11))   # fraction within each bin
    ax.figure.savefig("class_fractions.png")
    #   per_bin=False normalises to the total orbit count instead.

    # Bar chart of the orbit count per class:
    ax = cls.plot_class_histograms()
    ax.figure.savefig("class_histogram.png")
```

Compare two snapshots (e.g. before/after a perturbation) particle-by-particle,
matched by particle ID — particles in only one snapshot are dropped, and it is
up to you which snapshot is the earlier one:

```python
before = res_early.classify()
after = res_late.classify()

cmp = before.compare(after)          # `before` is the "before" state by convention
cmp.n_matched                        # particles present in both
cmp.fraction_changed                 # fraction that switched family
cmp.changed                          # (M,) bool, per matched particle (by cmp.ids)
rows, cols, matrix = cmp.transition_matrix()   # counts of before-class -> after-class

# Sankey diagram of the family flow from `this` (before) to `other` (after):
ax = cmp.plot_sankey()
ax.figure.savefig("family_flow.png")
```

Both classifications must use the same class scheme — condense both with
`condense_families()` first, or compare two full classifications.

### Figure rotation

`prepare()` also checks for **figure rotation** (a tumbling, non-axisymmetric
figure) and logs a `WARNING` if it is detected: the classifier integrates orbits
in a *static* potential, so a tumbling figure would produce erroneous orbit
families. Inspect the diagnostics directly with `ps.detect_figure_rotation()`
(returns the axis ratios, the rotation measure `|v_rot|/sigma`, and the short
axis), or skip the check with `ps.prepare(check_figure_rotation=False)`.

Single-snapshot detection is necessarily heuristic — it flags a non-axisymmetric
figure with significant ordered rotation about its short axis. Confirm the actual
pattern speed from consecutive snapshots before discarding a classification.

### Logging

lanfear logs through Python's `logging` module under the `"lanfear"` logger,
configured automatically on import (default level `WARNING`). Set the verbosity
from your script:

```python
import lanfear as lf

lf.set_verbosity("INFO")   # or "DEBUG", "WARNING", ..., or a logging.* integer
```

`INFO` reports the main pipeline steps (scale radius, potential build,
validation, integration timing, classification counts) and warns about failed
orbits; `DEBUG` adds finer detail (recentring, alignment, Gram-matrix
conditioning, black-hole parameters).

### Running in parallel (MPI)

Orbit integration is decomposed over MPI ranks; set `OMP_NUM_THREADS` for
per-rank threading (hybrid MPI+OpenMP).

```bash
# inside a SLURM allocation:
srun -n 64 python your_analysis.py
# on an interactive node:
mpirun -n 8 python your_analysis.py
# serial / debugging (no launcher needed):
python your_analysis.py
```

`scripts/run_orbits_mpi.py` is a runnable example and rank-count parity check.
mpi4py is required for parallel runs (`pip install mpi4py`); serial runs work
without it (the driver falls back automatically if MPI is unavailable).

### Progress reporting

Orbit integration is the dominant cost, so the C++ core prints
`"<X>% of particles integrated"` to the console at every 10% of orbits. This is
on by default for `integrate_family` / `analyse_family` (and
`integrate_states` / `analyse_states`); pass `progress=False` to silence it.
Under MPI only the root rank reports, on its own share of the orbits.

## Units

The SCF core works in Hernquist-Ostriker units: `G = M_field = scale_radius = 1`.
The scale radius is estimated from the field half-mass radius as
`r_half / (1 + sqrt(2))` (exact for a Hernquist profile). The Python layer
converts physical coordinates to/from HO units; black-hole masses and positions
are supplied in physical units and normalised internally.

## Layout

```
include/lanfear/   C++ headers (header-only physics)
  scf_potential.hpp    HO/SCF expansion + softened BH, fast recurrence eval
  disc_potential.hpp   Miyamoto-Nagai disc basis + softened BH
  orbit_integrator.hpp Boost-odeint orbit integration (templated on potential)
  frequency.hpp        radix-2 FFT + NAFF spectral-line extraction
  orbit_analysis.hpp   integrate + frequency-analyse per orbit
  special_functions.hpp Gegenbauer (Boost) + associated Legendre
  array3d.hpp          contiguous 3D coefficient store
src/
  python_bindings.cpp  pybind11 module -> lanfear/_core (incl. pickling)
  test_core.cpp        standalone C++ physics check (analytic Hernquist)
lanfear/           Python package
  particle_system.py   Gadget-4 HDF5 reader + preparation
  potential.py         HO Potential wrapper, unit handling, validation
  disc_potential.py    DiscPotential wrapper (MN basis, Gram solve, validation)
  orbits.py            MPI driver (integrate_family / analyse_family / ...)
  classify.py          orbit classification (families from summary + freqs)
scripts/
  run_orbits_mpi.py    runnable MPI example + rank-count parity check
tests/
  test_pipeline.py     Milestone 1: potential + validation
  test_orbits.py       Milestone 2: integration physics + MPI parity
  test_frequencies.py  Milestone 3: NAFF + frequency pipeline + MPI parity
  test_classify.py     Milestone 4: classification of known orbit types
  test_disc.py         Milestone 5: disc basis + disc pipeline
```

## References

- Hernquist & Ostriker 1992, ApJ 386, 375 (SCF / HO basis).
- Miyamoto & Nagai 1975, PASJ 27, 533 (disc density-potential pair).
- Laskar 1990; Valluri & Merritt 1998 (NAFF frequency analysis).
- Carpintero & Aguilar 1998, MNRAS 298, 1 (frequency-based classification).
