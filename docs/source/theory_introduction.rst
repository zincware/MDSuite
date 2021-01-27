MDSuite Theory
==============

There is a lot of theory that goes into the MDSuite package in order to perform the analysis available. In this part
of the documentation we will attempt to sufficiently cover the theoretical foundations of the methods used in the
package.

Before we get started we want to say that if you find any mistakes in this theory, or alternatively you would like to
see more information about a specific concept, feel free to contact us and let us know.

Physical Properties
-------------------
The goal of MDSuite is to calculate physical properties of systems from experiments. Broadly speaking we try to split
split analysis into two groups, namely structural, and dynamic properties of a material. As we add functionality to the
code the documentation will grow to incorporate the theoretical background for that calculation.

Dynamic Properties:
^^^^^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 1

    _theory/diffusion_coefficients
    _theory/ionic_conductivity
    _theory/viscosity
    _theory/thermal_conductivity

For each of these calculations we have implemented several types of calculations depending on what data is available in
the experiment. These two methods are often referred to as the Green-Kubo methods, where an autocorrelation function
is computed in accordance with the fluctuation-dissipation theorem, or the Einstein method, where a mean square
displacement of some property is calculated to describe the motion. In some cases, it is possible to compute these
properties using a pre-computed flux file rather than atomistic data. In these cases, we have also introduced functions
which can solve for these properties using such a flux file.

Structural Properties:
^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   _theory/radial_distribution_function
   _theory/coordination_numbers
   _theory/structure_factor
   _theory/kirkwood_buff_integrals
   _theory/phonon_spectrum
   _theory/potential_of_mean_force
   _theory/isothermal_compressibility

Each of these analysis have very similar interfaces and take similar arguments. In each case, if the arguments are not
changed, the analysis will return figures of the analysis and .npy data files stored inside the experiment directory.
Furthermore, any property value calculated during the analysis is stored in the class state and can be read out by
calling the relevant property from a python script.

Theoretical/Computational Concepts
----------------------------------
In addition to the discussion surrounding the properties being calculated and how that is performed, we also have some
more abstract theory sections relating either to computational methods or some broader theory which, whilst being
implemented in the package, is not directly related to a single calculation.

Computational Topics:
^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   _theory/tensor_operations_in_mdsuite
   _theory/hdf5_data_structure
   _theory/autocorrelation_in_mdsuite
   _theory/ensemble_average


Theoretical Concepts:
^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   _theory/fluctuation_dissipation_theorem
   _theory/green_kubo_relations
   _theory/einstein_calculations
   _theory/onsager_coefficients_and_dynamics_in_md_simulations
   _theory/diffusion_coefficients
   _theory/ionic_conductivity
   _theory/viscosity
   _theory/thermal_conductivity




Final Words
-----------
This theory portion of the documentation is a work in progress, so we would just like to ask one more time that if you
happen to find any errors or think something needs to be added, please do let us know and we an arrange to correct it.
