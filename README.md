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

Usage, configuration guides and the API reference are on the
[wiki](https://github.com/alex-rawlings/lanfear/wiki).

## Requirements

- Python ≥ 3.11 (developed against 3.13), NumPy, h5py, matplotlib.
- To build the C++ extension (`.[build]`): pybind11 ≥ 2.10, CMake ≥ 3.15,
  Boost ≥ 1.70.
- Optional: mpi4py for MPI-parallel orbit integration (`.[mpi]`); SciPy, ruff,
  pytest, pre-commit for development (`.[dev]`); Sphinx, furo to build the
  docs site (`.[docs]`, only needed if previewing it locally — see the
  [wiki](https://github.com/alex-rawlings/lanfear/wiki)).

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

## References

- Hernquist & Ostriker 1992, ApJ 386, 375 (SCF / HO basis).
- Miyamoto & Nagai 1975, PASJ 27, 533 (disc density-potential pair).
- Laskar 1990, Icarus 88, 266 (frequency-map analysis and diffusion rate); Valluri & Merritt 1998 (NAFF frequency analysis).
- Carpintero & Aguilar 1998, MNRAS 298, 1 (frequency-based classification).
- Frigo et al. 2021, MNRAS 508, 4610 (irregular/chaotic orbit classification).

[![Ruff PR Check](https://github.com/alex-rawlings/lanfear/actions/workflows/ruff.yml/badge.svg)](https://github.com/alex-rawlings/lanfear/actions/workflows/ruff.yml)
