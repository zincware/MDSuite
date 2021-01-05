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

* `Self diffusion coefficients <diffusion_coefficients.html>`_
* `Distinct diffusion coefficients <diffusion_coefficients.html>`_
* `Ionic conductivity <ionic_conductivity.html>`_
* `Viscosity <viscosity.html>`_
* `Thermal conductivity <thermal_conductivity.html>`_

For each of these calculations we have implemented several types of calculations depending on what data is available in
the experiment. These two methods are often referred to as the Green-Kubo methods, where an autocorrelation function
is computed in accordance with the fluctuation-dissipation theorem, or the Einstein method, where a mean square
displacement of some property is calculated to describe the motion. In some cases, it is possible to compute these
properties using a pre-computed flux file rather than atomistic data. In these cases, we have also introduced functions
which can solve for these properties using such a flux file.

Structural Properties:
^^^^^^^^^^^^^^^^^^^^^^

* `Radial distribution function <radial_distribution_function.html>`_
* `Coordination numbers <calculate_coordination_numbers.html>`_
* `Structure factor <structure_factor.html>`_
* `Kirkwood-Buff Integrals <kirkwood_buff_integrals.html>`_
* `Phonon spectrum <phonon_spectrum.html>`_
* `Potential of mean-force <potential_of_mean_force.html>`_
* `Isothermal compressibility <isothermal_compressibility.html>`_

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

* `Tensor operations in MDSuite <tensor_operations_in_mdsuite.html>`_
* `HDF5 data structure <hdf5_data_structure.html>`_
* `Autocorrelation in MDSuite <autocorrelation_in_mdsuite.html>`_

Theoretical Concepts:
^^^^^^^^^^^^^^^^^^^^^

* `The Fluctuation-Dissipation Theorem <fluctuation_dissipation_theorem.html>`_
* `Green-Kubo relations <green_kubo_relations.html>`_
* `Einstein type calculations <einstein_calculations.html>`_
* `Onsager coefficients and dynamics in MD simulations <onsager_coefficients_and_dynamics_in_md_simulations.html>`_

Final Words
-----------
This theory portion of the documentation is a work in progress, so we would just like to ask one more time that if you
happen to find any errors or think something needs to be added, please do let us know and we an arrange to correct it.