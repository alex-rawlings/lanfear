<p align="center">
<img src="logo/lanfear.png" width="300">
</p>

# LANFEAR: Light-weight Analytic N-body Field Expansion to Ascertain Resonances

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

- Python ≥ 3.11 (developed against 3.13), NumPy, h5py, matplotlib.
- To build the C++ extension (`.[build]`): pybind11 ≥ 2.10, CMake ≥ 3.15,
  Boost ≥ 1.70.
- Optional: mpi4py for MPI-parallel orbit integration (`.[mpi]`); SciPy, ruff,
  pytest, pre-commit for development (`.[dev]`); Sphinx, furo to build the
  docs site (`.[docs]`, only needed if previewing it locally — see
  [Documentation](#documentation)).

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
pip install -e ".[docs]"  # sphinx, furo -- only if previewing the docs site locally
pip install -e ".[all]"   # every optional dependency (mpi + build + dev + docs)
```

The compiled extension is ABI-specific to the Python it was built against, so build and install in the same environment; the wheel
is not portable across machines/Python versions.

## Usage

```python
import lanfear as lf

ps = lf.ParticleSystem.from_gadget_hdf5("snapshot.hdf5")
ps.prepare()             # recentre (shrinking sphere), align, scale radius, figure-rotation check

# Spherical-ish systems: Hernquist-Ostriker basis.
pot = lf.Potential.from_particles(ps, n_max=18, l_max=7)
# Flattened / disc-like systems: Miyamoto-Nagai disc basis (same interface).
#   pot = lf.DiscPotential.from_particles(ps, n_radial=10, n_vert=3)
result = pot.validate()                       # analytic potential vs direct sum
print(result)                                 # median / p90 / worst rel. error
assert result.passed(tolerance=0.02)

# Unsure how to pick n_max / l_max? Sweep them and watch the error converge:
sweep = lf.Potential.truncation_convergence(
    ps, n_max_values=[2, 6, 12, 18], l_max_values=[0, 2, 4, 6]
)
sweep.plot()                                  # median error vs n_max and vs l_max

# Evaluate (HO units) at physical coordinates:
phi = pot.potential([[1.0, 0.0, 0.0]])
acc = pot.acceleration([[1.0, 0.0, 0.0]])

# The potential is built from ALL particles, but integration can be restricted
# to a spatial region: `radius_mask` composes with `select` like `species_mask`.
inner = ps.select(ps.radius_mask(r_max=10.0))         # subset within r
res = lf.analyse_family(pot, inner, family="STAR")    # only inner stars integrated
#   combine masks: ps.select(ps.species_mask("STAR") & ps.radius_mask(10.0))

# Integrate every star for 50 orbital periods and frequency-analyse it
# (fundamentals + spectral lines per axis, in one pass; MPI-distributed if
# launched under srun/mpirun, otherwise serial):
res = lf.analyse_family(pot, ps, family="STAR", n_periods=50, n_lines=4)
if res is not None:                           # None on non-root MPI ranks
    print(res.column("energy_drift"))         # per-orbit summary columns
    print(lf.SUMMARY_COLUMNS)                 # available quantities
    good = res.ok                             # status == 0

    res.fundamentals        # (N, 3) signed fundamental frequency per axis (HO)
    res.lines               # (N, 3, n_lines, 2) leading (freq, amp) per axis
    res.frequency_ratios    # (N, 2) |w_x|/|w_z|, |w_y|/|w_z|

    # Integration is expensive -- save the results and reload later without
    # re-integrating (a compact .npz holding everything OrbitResults needs):
    res.save("orbits.npz")
    res = lf.OrbitResults.load("orbits.npz")   # resume classification/plotting

    # Classify into orbit families (pi-box / tube / rosette / boxlet /
    # irregular ...). An orbit whose spectrum needs > 3 base frequencies is
    # labelled `irregular` (Frigo et al. 2021) -- a likely-chaotic candidate.
    cls = res.classify()
    cls.labels              # (N,) lf.OrbitClass values
    cls.names               # (N,) family name strings
    cls.counts()            # {family_name: count}
    z_tubes = cls.mask(lf.OrbitClass.SHORT_AXIS_TUBE)
    chaotic = cls.mask(lf.OrbitClass.IRREGULAR)
    z_tube_ids = cls.get_class_ids(lf.OrbitClass.SHORT_AXIS_TUBE)  # particle IDs

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

    # Frequency map: scatter of (|w_x|/|w_z|, |w_y|/|w_z|) coloured by class,
    # where regular families cluster and resonances trace straight lines:
    ax = cls.plot_frequency_map()
    ax.figure.savefig("frequency_map.png")
```

Compare two snapshots (e.g. before/after a perturbation) particle-by-particle,
matched by particle ID — particles in only one snapshot are dropped, and it is
up to you which snapshot is the earlier one:

```python
before = res_early.classify()
after = res_late.classify()

cmp = before.compare(after)          # `before` is the "before" state by convention
cmp.n_matched                        # particles present in both
cmp.fraction_changed                 # fraction that switched family at any stage
cmp.changed                          # (M,) bool, per matched particle (by cmp.ids)
rows, cols, matrix = cmp.transition_matrix()   # counts of before-class -> after-class

# Sankey diagram of the family flow from `this` (before) to `other` (after):
ax = cmp.plot_sankey()
ax.figure.savefig("family_flow.png")
```

`compare()` also accepts an iterable of classifications to compare more than two
snapshots in sequence (`self` followed by each item, in order), producing one
stage per classification — e.g. `before.compare([mid, after])`. `transition_matrix(stage=i)`
then indexes the `i`-th consecutive pair, and `plot_sankey()` draws every stage
as one multi-column diagram.

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

### Plotting a single trajectory

Storing every particle's full phase-space trajectory "just in case" is wasteful
when only a handful are ever inspected in detail, so `ParticleTrajectory`
re-integrates just the one orbit you ask for (rather than reading it back out
of a batch `analyse_family`/`analyse_states` run) and plots it:

```python
traj = lf.ParticleTrajectory.from_particles(pot, ps, particle_id=12345, n_periods=10)
# or, without a ParticleSystem, from a physical position/velocity directly:
#   traj = lf.ParticleTrajectory.integrate(pot, pos_phys, vel_phys, n_periods=10)

axes = traj.plot()              # x-y, x-z, y-z projections, coloured by time (BuPu)
axes[0].figure.savefig("trajectory.png")
```

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

### Choosing n_max / l_max

`scripts/sweep_truncation.py` sweeps a grid of `(n_max, l_max)`, building and
validating the potential at each point (one MPI rank per combination), then
recommends the cheapest orders at the convergence knee — feed the result to
`run_orbits_mpi.py`:

```bash
srun --mpi=pmix -n 16 python scripts/sweep_truncation.py --file snap.hdf5
# -> Recommended: n_max=8 l_max=4   ->  run_orbits_mpi.py --n-max 8 --l-max 4
```

It prints the full error grid and can save a heatmap with `--plot sweep.png`. The
recommendation is convergence-relative (cheapest orders within `--slack` of the
best grid error), *not* an absolute median-error cut — the median potential error
is monopole-dominated, so an absolute cut would wrongly accept `l_max = 0`.

### Progress reporting

Orbit integration is the dominant cost, so the C++ core prints
`"<X>% of particles integrated"` to the console at every 10% of orbits. This is
on by default for `analyse_family` (and `analyse_states`); pass
`progress=False` to silence it. Under MPI only the root rank reports, on its
own share of the orbits.

## Units

The SCF core works in Hernquist-Ostriker units: `G = M_field = scale_radius = 1`.
The scale radius is estimated from the field half-mass radius as
`r_half / (1 + sqrt(2))` (exact for a Hernquist profile). The Python layer
converts physical coordinates to/from HO units; black-hole masses and positions
are supplied in physical units and normalised internally.

## Documentation

API documentation is built with Sphinx from the docstrings under `lanfear/`
(source in `docs/`). On every push to `main`, `.github/workflows/docs.yml`
renders it to Markdown and pushes it to this repo's
[wiki](https://github.com/alex-rawlings/lanfear/wiki), one page per module
(`particle_system`, `potentials`, `orbits`, `classification`, `logging`).
`Home.md` (and `_Sidebar.md`, if present) is hand-maintained on GitHub and
never touched by this workflow.

To build it locally:

```bash
pip install -e ".[docs]"
cd docs
make html   # -> docs/_build/html/index.html (multi-page; for local browsing)
```

## Layout

```
include/lanfear/   C++ headers (header-only physics)
  centring.hpp         shrinking-sphere centre (position + bulk velocity)
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
  orbits.py            MPI driver (analyse_family / analyse_states / OrbitResults);
                       single-particle ParticleTrajectory (integrate + plot)
  classify.py          orbit classification (families from summary + freqs)
scripts/
  run_orbits_mpi.py    runnable MPI example + rank-count parity check
  sweep_truncation.py  (n_max, l_max) grid sweep + order recommendation
docs/
  conf.py                        Sphinx config (html locally, markdown for the wiki)
  index.rst + one .rst per module   API documentation sources (autodoc + napoleon)
  tidy_markdown.py               Parameters -> tables, fixes GitHub's wiki anchor slugs
tests/
  test_pipeline.py        Milestone 1: potential + validation
  test_orbits.py          Milestone 2: integration physics + MPI parity
  test_frequencies.py     Milestone 3: NAFF + frequency pipeline + MPI parity
  test_classify.py        Milestone 4: classification of known orbit types
  test_disc.py            Milestone 5: disc basis + disc pipeline
  test_figure_rotation.py detect_figure_rotation() heuristic
```

## References

- Hernquist & Ostriker 1992, ApJ 386, 375 (SCF / HO basis).
- Miyamoto & Nagai 1975, PASJ 27, 533 (disc density-potential pair).
- Laskar 1990; Valluri & Merritt 1998 (NAFF frequency analysis).
- Carpintero & Aguilar 1998, MNRAS 298, 1 (frequency-based classification).
- Frigo et al. 2021, MNRAS 508, 4610 (irregular/chaotic orbit classification).
