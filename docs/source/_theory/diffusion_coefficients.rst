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

Distinct Diffusion Coefficients
-------------------------------
If there is a need to better understand ionic correlation between the constituents in a system it can be useful to
calculate the distinct diffusion coefficients. These coefficients are calculated by tracking the motion of an atom
through the bulk with respect to all of the other atoms in the solution.