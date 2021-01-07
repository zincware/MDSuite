:orphan:
Diffusion Coefficients
======================

Diffusion coefficients are a dynamic property describing the motion of the constituents of a system. In the case of a
purely atomistic system, this is just the motion of the ions through the bulk. If molecules are present in the solution,
it is also possible to calculate the diffusion coefficients of the molecule. In MDSuite, we have two different types of
diffusion coefficients, the self diffusion, and the distinct diffusion coefficients.

Self Diffusion Coefficients
---------------------------
If we are interested solely in the diffusion of a single atomic species through the bulk we can calculate the self
diffusion coefficients. This can be described as the motion of an atom through the bulk with respect to itself.
There are two ways for us to calculate the self-diffusion coefficients from a simulation, the Einstein approach, and
the Green-Kubo approach. Both of these methods whilst equivalent, find use under different circumstances which we will
discuss here as well.

Einstein Method
***************
The Einstein method may be derived from the general Einstein-Smoluchowsky equation for for the trajectory of a Brownian
particle, for brevity, we state here only the result as

.. math::

    \langle|\mathbf{r}(t + \tau) - \mathbf{r}(t)|^{2}\rangle = 2dD\tau

Where :math:`\mathbf{r}(t)` is the position of a particle at time t, :math:`\tau` is a shift forwards in time,
:math:`d` is the dimensionality of the system, and :math:`D` is the diffusion coefficient. The angled brackets denote
an ensemble average over a trajectory, which is to say one should perform this averaging over different sets of data. In
practice, this is done by selecting a time range over which the analysis will be performed, and then performing it over
this time starting at different initial configurations.

In a typical MD simulation we will have more than a single particle in a system. In these cases, this calculation should
be performed on all the atoms and then averaged. Of course, in the case where a single atom is being studied, this is
not true. For a many body simulation, we modify the equation above to

.. math::

    D_{\alpha} = \frac{1}{2dD\tau}\sum_{i=0}^{N_{\alpha}}\langle|\mathbf{r}_{i}(t + \tau) - \mathbf{r}_{i}(t)|^{2}\rangle

where the summation goes over all particle of type :math:`\alpha` in the system. In this way, the average value for the
diffusion of this type is calculated.

Green-Kubo Method
*****************
The next method of calculation is the Green-Kubo approach, which, as mentioned above, can be derived from the Einstein
method, and vice-versa. We will save a discussion of the general Green-Kubo approach to calculations for the
`Green-Kubo Relations <green_kubo_relations.html>`_ section. Here, we simply write the diffusion out as

.. math::

    D = \frac{1}{N_{\alpha}d}\int d\tau \langle \sum_{i}^{N_{\alpha}}\mathbf{v}_{i}(t+\tau) \cdot \mathbf{v}_{i}(t) \rangle

where :math:`\mathbf{v}(t)` is the velocity of a particle at time t. In principle, the summation over atoms can be done
at several points of the calculation. In the case of MDSuite, we perform the summation for each part of the ensemble
average and integrate over the average over atoms. This is so that a nice smooth and accurate function may be integrated
over, and then several of these can be averaged to get a final diffusion coefficient with a reasonable estimate for error.

Which One Should I Use?
***********************
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


Distinct Diffusion Coefficients
-------------------------------
If there is a need to better understand ionic correlation between the constituents in a system it can be useful to
calculate the distinct diffusion coefficients. These coefficients are calculated by tracking the motion of an atom
through the bulk with respect to all of the other atoms in the solution.