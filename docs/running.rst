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

``scripts/run_orbits_mpi.py`` is a runnable example and rank-count parity check.
mpi4py is required for parallel runs (``pip install mpi4py``); serial runs work
without it (the driver falls back automatically if MPI is unavailable).

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

``INFO`` reports the main pipeline steps (scale radius, potential build,
validation, integration timing, classification counts) and warns about failed
orbits; ``DEBUG`` adds finer detail (recentring, alignment, Gram-matrix
conditioning, black-hole parameters). See :doc:`logging` for the API.
