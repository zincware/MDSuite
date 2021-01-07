:orphan:
Radial Distribution Function
============================
The radial distribution function (RDF) is one of the most commonly used properties in molecular dynamics (MD)
simulations to assess the structure of the system being studied.
A partial radial distribution function is a describes the probability of finding a particle of species :math:`beta`
a distance of r from species :math:`\alpha`.
Mathematically one can write this as

.. math::
    g_{ab}(r) = \rho 4 \pi r^{2} dr \sum_{i=1}^{N_a} \sum_{j=1}^{N_b}
            \langle \delta(|\mathbf{r}_i - \mathbf{r}_j| - r_{c}) \rangle

Implementation
--------------
MDSuite differs from other software in the calculation of the radial distribution function.
Rather than looping over atomic positions one species at at time, Tensorflow is used to calculate distances between
all atoms in a single tensor.
This tensor may then be operated on in order to calculate the RDF.