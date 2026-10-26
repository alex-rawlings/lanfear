Running and monitoring
======================

Running in parallel (MPI)
-------------------------

Orbit integration is decomposed over MPI ranks; set ``OMP_NUM_THREADS`` for
per-rank threading (hybrid MPI+OpenMP).

.. code-block:: bash

   # inside a SLURM allocation:
   srun -n 64 python your_analysis.py
   # on an interactive node:
   mpirun -n 8 python your_analysis.py
   # serial / debugging (no launcher needed):
   python your_analysis.py

SLURM writes the job log itself, before Python starts, so where it goes is set in
the batch script rather than in ``run_orbits_mpi.py``. Send it to a
``slurm_output`` subdirectory with ``#SBATCH --output=slurm_output/slurm-%j.out``
(``%j`` is the job ID). SLURM does not create the directory, so run
``mkdir -p slurm_output`` in the submission directory before ``sbatch``.

``scripts/run_orbits_mpi.py`` is a runnable example and rank-count parity check.
mpi4py is required for parallel runs (``pip install mpi4py``); serial runs work
without it (the driver falls back automatically if MPI is unavailable). Its
``--pattern-speed`` option sets the figure rotation. It takes ``estimate``
(the default), ``none`` for a static potential, or a number. For a rotating
figure, ``--max-body-period-factor`` (default 8) caps how much longer orbits
near corotation are integrated to span ``--periods`` body-frame periods, and a
summary of the lengthening and the runaways is printed. ``--runaway-factor``
sets the runaway flag. Adaptive extension of drifting orbits
(``--max-extensions``, see :doc:`tuning`) is off by default, as in
``analyse_family``. See "Figure rotation" in :doc:`preparation`;
``scripts/trajectory.py`` takes the same ``--pattern-speed`` option.
``--ids FILE`` (a ``.npy`` array, or whitespace-separated text) loads only the
listed particle IDs from ``--file``, for snapshots too large to hold in memory
such as cosmological boxes. The snapshot is streamed in chunks and only the
matching particles are kept, so the potential is built from those particles
alone. The list should therefore hold the whole system (e.g. every member of
one halo, with its black holes if centring on them). The default loads every
particle. A periodic snapshot is unwrapped across the box edges, so a halo
straddling one loads contiguous; ``--periodic auto`` (the default) detects this
from Gadget-4's ``PERIODIC`` option, and ``yes``/``no`` override it.
Particle data are always converted to physical units on loading: a
cosmological snapshot is detected from Gadget-4's ``ComovingIntegrationOn``
parameter (``--cosmological auto``, the default; ``yes``/``no`` override it)
and its comoving, *h*-scaled positions, velocities and masses are rescaled with
the scale factor and ``HubbleParam``. An isolated run is already physical and
is left as stored. Gadget-3 snapshots have no ``Parameters`` or ``Config``
group, so ``auto`` falls back to the header: a run is comoving if
``Redshift = 1/Time - 1`` and ``Omega0 > 0`` (Gadget-3 writes ``Redshift = 0``
for non-comoving runs), and a comoving snapshot with a positive ``BoxSize`` is
periodic.
``--centre auto`` (the default) centres on the black hole(s) if the snapshot has
any, otherwise with the shrinking sphere (the built-in test snapshot has none).
The snapshot is loaded, centred and fitted on the root rank only. If that fails
(for example the potential misses the 0.1% accuracy target), the failure is
broadcast and every rank exits with an error, rather than the other ranks
waiting for orbits that never arrive.

Under MPI, ``analyse_family`` shuffles the orbits with a fixed permutation
before splitting them across ranks and restores the input order after the
gather. Snapshots are usually stored in spatial (Peano-Hilbert) order, so
unshuffled contiguous blocks would be patches of the galaxy with very
different costs (most strongly near corotation, where orbits are integrated
for longer), and one rank could be left with several times the average work.
The results do not depend on the permutation. Note that the root rank's
progress messages count only its own share of the orbits.

Progress reporting
------------------

Orbit integration is the dominant cost, so the C++ core prints
``"<X>% of particles integrated"`` to the console at every 10% of orbits. This is
on by default for ``analyse_family`` (and ``analyse_states``); pass
``progress=False`` to silence it. Under MPI only the root rank reports, on its
own share of the orbits.

Logging
-------

lanfear logs through Python's ``logging`` module under the ``"lanfear"`` logger,
configured automatically on import (default level ``WARNING``). Set the
verbosity from your script:

.. code-block:: python

   import lanfear as lf

   lf.set_verbosity("INFO")   # or "DEBUG", "WARNING", ..., or a logging.* integer

``INFO`` reports the main pipeline steps (scale radius, estimated pattern speed,
potential build, SMBH-binary properties and how the binary was attached, validation,
integration timing, the number of binary-interacting orbits, classification
counts) and warns about failed orbits; ``DEBUG`` adds finer detail (recentring, alignment, Gram-matrix
conditioning, black-hole parameters). See :doc:`logging` for the API.
