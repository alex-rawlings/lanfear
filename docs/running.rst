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
(the default), ``none`` for a static potential, a number (about the aligned
short axis) or an ``x,y,z`` vector. For a rotating figure,
``--max-body-period-factor`` (default 8) caps how much longer orbits near
corotation are integrated to span ``--periods`` body-frame periods,
``--runaway-factor`` sets the runaway flag, and a summary of the pattern-speed
ratio, the near-corotation fraction and the runaways is printed. A
non-default ``--pattern-speed`` is recorded in the output file name
(e.g. ``orbits_STAR_ps-none_snap.npz``). ``--centre auto`` (the default)
centres on the black hole(s) if there are any, otherwise with the shrinking
sphere. See "Figure rotation" in :doc:`preparation`; ``scripts/trajectory.py``
takes the same ``--pattern-speed`` and ``--centre`` options.

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
