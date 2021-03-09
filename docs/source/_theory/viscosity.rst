Viscosity
=====================================================

The viscosity :math:`\mu` of a fluid refers to its resistance to be deformed at a given rate.
In MDSuite, this property can be computed in two different ways:
i) :ref:`_theory/green_kubo_relations:Green-Kubo Relations` and ii) :ref:`_theory/einstein_calculations:Einstein Relations`.
These two methods are based on Equilibrium Molecular Dynamics (EMD).

Green-Kubo
---------------------------
Assuming that there is not a preferential direction, the equation to compute viscosity by means of Green-Kubo reads:

.. math::


    \kappa = \frac{V}{3 k_B T} \int \langle P_{a,b}(0) \cdot P_{a,b}(t) \rangle \mathrm{d} t \qquad (a \neq b \in \{x,y,z\})

where :math:`V` is the volume of the system, :math:`k_b` is the Boltzman constant, :math:`T` is the temperature and
:math:`P_{a,b}(t)` are the off-diagonal components of the stress tensor at time :math:`t`.

The :math:`P_{a,b}(t)` can be computed from the contributions of each atom as:

.. math::

    P_{a,b} = \frac{\sum_{i=1}^{N_{atoms}} m^i v_{a}^i v_{b}^i}{V} + \frac{\sum_{i=1}^{N_{atoms}} r_{a}^i f_{b}^i}{V}

And, actually, the two contributions from the summations are equal to minus the stress tensor, so the previous expression can be written as: 

.. math::

    P_{a,b}\cdot V = \sum_{i=1}^{N_{atoms}} m^i v_{a}^i v_{b}^i + \sum_{i=1}^{N_{atoms}} r_{a}^i f_{b}^i = -\sum_{i=1}^{N_{atoms}} \mathbf{S}_{a,b}^i

:math:`\mathbf{S}_{i,a,b}` is the off-diagonal components of the stress tensor of atom :math:`i`. This quantity can be easily extracted from MD codes. 

In principle, the summation over atoms can be done
at several points of the calculation. In the case of MDSuite, we perform the summation for each part of the ensemble
average and integrate over the average over atoms. This is so that a nice smooth and accurate function may be integrated
over, and then several of these can be averaged to get a final diffusion coefficient with a reasonable estimate for error.
We will save a discussion of the general Green-Kubo approach to calculations for the
:ref:`_theory/green_kubo_relations:Green-Kubo Relations` section.

When using trajectory files, the user must provide the aforementioned quantities. However, for flux files, the off-diagonal components of the pressure tensor can be printed directly. 

Einstein-Helfand
---------------------------

Einstein-Helfand method can be used to compute the thermal conductivity :footcite:`kinaci_calculation_2012`:

.. math::

    \kappa = \frac{1}{V k_B T} \lim_{t \to \infty} \frac{1}{2t} \langle [L_{a,b}^i(t)-L_{a,b}^i(0)]^2  \rangle

where :math:`L_{a,b}(t)` is defined as

.. math::

    L_{a,b}^i(t) = p_{a}^i(t) \cdot r^i_b(t)

where :math:`p_{a}^i` and :math:`r_{b}(t)` are the momentum and position of particle :math:`i`.
The angled brackets denote
an ensemble average over a trajectory, which is to say one should perform this averaging over different sets of data. In
practice, this is done by selecting a time range over which the analysis will be performed, and then performing it over
this time starting at different initial configurations

Which One Should I Use?
---------------------------
Great question, and it is totally dependant on what simulation results you have access to. In order for the Green-Kubo
calculations to work well, you will need to have atomic configurations spaced relatively close together. This is because
the calculation measures correlation with time, if the sample rate is too large, the finer details for this correlation
will be missed. However, in the case of the Green-Kubo formulation, there are no errors induced by fitting to a line, and
can therefore be a good starting point or sanity check during analysis.
For long simulations on big systems, it is more storage efficient to store configurations less often. In these cases,
you will need to use the Einstein approach as it is far less susceptible to poor resolution.
If you aren't sure, perform both and take a look at the plots. An autocorrelation function with limited resolution will
be fairly obvious. The same will go for looking at the fit of the Einstein method. Ideally, and this goes for the test
cases provided in the MDSuite documentation, you will be able to match the Einstein and Green-Kubo methods to each other
for a complete sanity check, but this is only possible for certain, often very fast (speaking to the dynamics of the
particles) systems.

.. footbibliography::
