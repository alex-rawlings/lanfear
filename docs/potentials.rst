Potentials
==========

``Potential``, ``DiscPotential`` and ``MultiComponentPotential`` share a common
interface -- unit handling (``to_ho_state``, ``period_to_physical``), black-hole
bookkeeping (``add_black_hole``, ``n_black_holes``), evaluation (``potential``,
``acceleration``, ``core`` for the orbit drivers), ``validate`` and
``plot_potential_plane`` -- implemented once and listed below on each class.

.. automodule:: lanfear.potential
   :members:
   :inherited-members:
   :show-inheritance:

.. automodule:: lanfear.disc_potential
   :members:
   :inherited-members:
   :show-inheritance:

.. automodule:: lanfear.multi_component_potential
   :members:
   :inherited-members:
   :show-inheritance:
