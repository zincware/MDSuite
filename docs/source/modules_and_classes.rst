Modules and Classes
===================

The code developed for MDSuite is quite broad and extensive.
In the sections outlined below you will find the sphinx documentation for the classes
and modules that have been developed for the analysis programs.
We have tried to split it up as much as possible but in doing so may have introduced
some human error.
The full API tree is also provided which will conatin all of the modules.

Full API Tree
-------------

.. toctree::
   :maxdepth: 1

   _modules/modules

Software architecture
---------------------

.. toctree::
   :maxdepth: 1

   _modules/modules
   _modules/mdsuite.project.project
   _modules/mdsuite.experiment.experiment
   _modules/mdsuite.memory_management
   _modules/mdsuite.experiment.run_module

Database Information
--------------------

.. toctree::
   :maxdepth: 1

   _modules/mdsuite.database.data_manager
   _modules/mdsuite.database.simulation_database

Calculators
-----------

.. toctree::
   :maxdepth: 1

   _modules/mdsuite.calculators.einstein_diffusion_coefficients
   _modules/mdsuite.calculators.einstein_helfand_ionic_conductivity
   _modules/mdsuite.calculators.green_kubo_self_diffusion_coefficients
   _modules/mdsuite.calculators.green_kubo_ionic_conductivity
   _modules/mdsuite.calculators.green_kubo_thermal_conductivity
   _modules/mdsuite.calculators.einstein_helfand_thermal_conductivity
   _modules/mdsuite.calculators.einstein_helfand_thermal_conductivity_kinaci
   _modules/mdsuite.calculators.kirkwood_buff_integrals
   _modules/mdsuite.calculators.nernst_einstein_ionic_conductivity
   _modules/mdsuite.calculators.potential_of_mean_force
   _modules/mdsuite.calculators.radial_distribution_function
   _modules/mdsuite.calculators.structure_factor

Transformations
---------------

.. toctree::
    :maxdepth: 1

    _modules/mdsuite.transformations.integrated_heat_current
    _modules/mdsuite.transformations.ionic_current.rst
    _modules/mdsuite.transformations.kinaci_integrated_heat_current.rst
    _modules/mdsuite.transformations.map_molecules.rst
    _modules/mdsuite.transformations.momentum_flux.rst
    _modules/mdsuite.transformations.scale_coordinates.rst
    _modules/mdsuite.transformations.thermal_flux.rst
    _modules/mdsuite.transformations.transformations.rst
    _modules/mdsuite.transformations.translational_dipole_moment.rst
    _modules/mdsuite.transformations.unwrap_coordinates.rst
    _modules/mdsuite.transformations.unwrap_via_indices.rst
    _modules/mdsuite.transformations.wrap_coordinates.rst
