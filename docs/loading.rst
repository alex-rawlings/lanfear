Loading snapshots
=================

:meth:`~lanfear.ParticleSystem.from_gadget_hdf5` reads a Gadget-3 or Gadget-4
HDF5 snapshot (single file) into a :class:`~lanfear.ParticleSystem`:

.. code-block:: python

   import lanfear as lf

   ps = lf.ParticleSystem.from_gadget_hdf5("snapshot.hdf5")

Every ``PartTypeX`` group with ``Coordinates`` is read, taking ``Coordinates``,
``Velocities``, ``ParticleIDs`` and the masses (per-particle ``Masses`` if
present, otherwise the constant from the header ``MassTable``). Each group gets
a species label:

=============  =========
Group          Species
=============  =========
``PartType0``  ``GAS``
``PartType1``  ``DM``
``PartType2``  ``DISK``
``PartType3``  ``BULGE``
``PartType4``  ``STAR``
``PartType5``  ``BH``
=============  =========

Whatever the snapshot, the loaded particles are contiguous (unwrapped across
the edges of a periodic box) and in physical units, so the rest of the pipeline
(:doc:`preparation`, the potential fit, the orbit integration) treats a halo cut
from a cosmological box exactly like an isolated galaxy.

Selecting particles by ID
-------------------------

A cosmological snapshot can hold billions of particles, too many to load. Pass
``ids`` to load only the particles of one system:

.. code-block:: python

   halo_ids = np.load("halo_ids.npy")
   ps = lf.ParticleSystem.from_gadget_hdf5("snap_099.hdf5", ids=halo_ids)

The snapshot is streamed in blocks of ``chunk_size`` rows (default ``2**22``).
The ``ParticleIDs`` of each block are read first, and the other datasets only
for the matching rows, so memory holds the selection plus one block, never the
whole snapshot. Particles keep their snapshot order, duplicate requests are
ignored, and requested IDs absent from the snapshot are reported in a warning
(an error if none is found).

* ``ids`` are the values of the ``ParticleIDs`` datasets, **not** positions in
  the per-type arrays. Halo finders often store the latter: e.g. Caesar's
  ``glist``/``slist``/``bhlist`` are indices into the gas, star and BH arrays.
  Passing such indices as IDs selects unrelated particles from across the box;
  a selection spanning more than half the box triggers a warning (see
  `Periodic boxes`_), which is the usual symptom.
* The potential is built from the loaded particles alone, so the list should
  hold the whole system: every species that contributes to the potential, and
  the black holes if centring on them (``centre="bh"``).
* Some codes give several particles the same ID (in SIMBA a star can share the
  ID of the gas particle it formed from); every particle with a requested ID is
  loaded.

Physical units
--------------

Particle data are always converted to physical units on loading. For a
cosmological snapshot, with scale factor ``a`` (header ``Time``) and Hubble
parameter ``h`` (``HubbleParam``), each field is multiplied by
``a**a_scaling * h**h_scaling``. Gadget-4 records the exponents on each
dataset; Gadget-3 records none, and the standard ones are used:

==================  =====================================  =============
Field               Stored as                              ``(a, h)``
==================  =====================================  =============
``Coordinates``     comoving length / h                    ``(1, -1)``
``Velocities``      ``v_pec / sqrt(a)``                    ``(0.5, 0)``
``Masses``          mass / h                               ``(0, -1)``
==================  =====================================  =============

The ``MassTable`` masses and ``BoxSize`` are converted in the same way. The
velocities are physical peculiar velocities, without the Hubble flow, as suits
orbits integrated in a fixed potential. Units otherwise stay those of the
simulation; lanfear's gravitational constant assumes the Gadget defaults (kpc,
1e10 Msun, km/s). The scale factor is kept in ``ps.scale_factor`` (``None``
for a non-cosmological snapshot).

A non-cosmological (isolated) run is already physical and is left exactly as
stored: its ``Time`` is a time, and ``HubbleParam`` plays no part in its
dynamics even when the parameter file sets it.

Whether a snapshot is cosmological is detected as follows; pass
``cosmological=True``/``False`` to override it:

* **Gadget-4**: the ``ComovingIntegrationOn`` parameter in the ``Parameters``
  group.
* **Gadget-3** (no ``Parameters`` group): the header must have
  ``Redshift = 1/Time - 1`` and ``Omega0 > 0``. Gadget-2/3 write
  ``Redshift = 0`` for non-comoving runs. ``Time = 1`` with ``Redshift = 0``
  fits either kind of run; it is taken as cosmological, with a warning.

Periodic boxes
--------------

A halo near the edge of a periodic box is split across it in the stored
coordinates. For a periodic snapshot each axis is wrapped to within half a box
of the mass-weighted circular mean of the loaded particles, so the halo comes
out contiguous (velocities are unchanged). The selection must therefore be
smaller than half the box; a warning is logged otherwise. The physical box side
lengths are kept in ``ps.box_size`` (``None`` for a non-periodic snapshot).

Periodicity is detected as follows; pass ``periodic=True``/``False`` to
override it:

* **Gadget-4**: the ``PERIODIC`` option in the ``Config`` group.
* **Gadget-3** (no ``Config`` group): a cosmological snapshot with a positive
  ``BoxSize`` is periodic. An isolated run is never treated as periodic, even
  if its header carries a ``BoxSize``.

Parallel reading
----------------

``n_workers`` reads the blocks in that many processes:

.. code-block:: python

   ps = lf.ParticleSystem.from_gadget_hdf5("snap.hdf5", ids=halo_ids, n_workers=16)

The result is identical to a serial read. HDF5 is not thread-safe, and h5py
serialises every call behind one lock, so the workers are separate processes,
each with its own read-only handle on the file; they share no mutable state,
and the blocks are reassembled in snapshot order whatever order they finish in.
The workers are spawned rather than forked (a fork would copy any lock held by
another thread of the parent, e.g. HDF5's, OpenMP's or MPI's, into the child,
where it is never released), so each starts by importing lanfear, which takes
about a second.

As a guide, selecting 2e5 particles from a 458 GB SIMBA snapshot (2.1e9
particles, uncompressed) took 93-162 s serially and 15.5 s with 16 workers, a
6x speed-up over the warm-cache serial read. 32 workers were slower than 16, as
the file system saturated, and memory stayed below 0.5 GB. Some points to
consider:

* The gain is largest for ``ids`` selections from large files, especially on
  parallel or network file systems and with compressed datasets.
* A full load (no ``ids``) also copies every particle from the workers back to
  the main process, so it gains less, and may be slower than a serial read for
  a small file on a fast local disk.
* Each worker holds up to one block, and up to ``2 * n_workers`` read blocks
  await reassembly, so peak memory grows by about ``3 * n_workers`` blocks.
* Use at most as many workers as the CPUs available to the loading process
  (e.g. its SLURM ``--cpus-per-task``).

Command line
------------

``scripts/run_orbits_mpi.py`` exposes the same options; the snapshot is loaded
on the root rank only, so ``--load-workers`` should match the CPUs of that rank:

.. code-block:: bash

   python scripts/run_orbits_mpi.py --file snap_099.hdf5 --ids halo_ids.npy \
       --load-workers 16

* ``--ids FILE``: a ``.npy`` array, or a text file of whitespace-separated IDs
  (``#`` starts a comment). The default loads every particle.
* ``--load-workers N``: processes reading the snapshot (default 1, serial).
* ``--periodic {auto,yes,no}`` and ``--cosmological {auto,yes,no}``: override
  the detection described above (default ``auto``).

Limitations
-----------

* Multi-file snapshots (``snap_099.0.hdf5``, ``snap_099.1.hdf5``, ...) are not
  read yet.
* A selection wider than half the box cannot be unwrapped unambiguously.
* No unit-system conversion is made beyond removing ``a`` and ``h``: a run in
  e.g. Mpc/h comes out in Mpc, which lanfear's default gravitational constant
  does not assume.
