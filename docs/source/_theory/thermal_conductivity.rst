Thermal conductivity
=====================================================

The thermal conductivity :math:`\kappa`  refers to the ability of a material to transfer or conduct heat. In an atomistic scale, heat conduction
is carried by phonons, electrons and photons. In the case of non-metallic materials, the transport through phonons is dominant, and
it is the one simulated in traditional MD.
In MDSuite, this property can be computed in two different ways:
i) :ref:`_theory/green_kubo_relations:Green-Kubo Relations` formulation and ii) :ref:`_theory/einstein_calculations:Einstein Relations` method.
These two methods are based on Equilibrium Molecular Dynamics (EMD).

Green-Kubo
---------------------------
Assuming that there is not a preferential direction, the equation to compute thermal conductivity by means of Green-Kubo reads:

.. math::

    \kappa = \frac{V}{3 k_B T^2} \int \langle \mathbf{J}(0) \cdot \mathbf{J}(t) \rangle \mathrm{d} t

where :math:`V` is the volume of the system, :math:`k_b` is the Boltzman constant, :math:`T` is the temperature and
:math:`\mathbf{J}` is the heat-flux at time :math:`t`. The heat-flux term :math:`\mathbf{J}` is computed as follows:

.. math::

    \mathbf{J}(t) = \frac{1}{V} \left[ \sum_{i=1}^{N_{atoms}} e_i \vec{v}_i - \sum_{i=1}^{N_{atoms}} \mathbf{S}_i \vec{v}_i \right]

in this case, :math:`e_i`,  :math:`\vec{v}_i` and  :math:`\mathbf{S}_i` are the energy, the velocity vector and
the stress tensor of atom :math:`i`, respectively. The energy of the atom computed internally by MDSuite adding the contributions of potential and kinetic.

In principle, the summation over atoms can be done
at several points of the calculation. In the case of MDSuite, we perform the summation for each part of the ensemble
average and integrate over the average over atoms. This is so that a nice smooth and accurate function may be integrated
over, and then several of these can be averaged to get a final diffusion coefficient with a reasonable estimate for error.
We will save a discussion of the general Green-Kubo approach to calculations for the
`Green-Kubo Relations <green_kubo_relations.html>`_ section.

When using trajectory files, the user must provide the aforementioned quantities. However, for flux files, the heat-flux :math:`\mathbf{J}`
should be pre-computed. This is typically done by the MD simulation code.

Einstein-Helfand
---------------------------

Einstein-Helfand method can be used to compute the thermal conductivity :footcite:`kinaci_calculation_2012`:

.. math::

    \kappa = \frac{1}{V k_B T^2} \lim_{t \to \infty} \frac{1}{2t} \langle [\mathbf{R}(t)-\mathbf{R}(0)]\cdot[\mathbf{R}(t)-\mathbf{R}(0)]  \rangle

Where :math:`\mathbf{R}(t)` is the integrated heat-flux at time :math:`t`. The angled brackets denote
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