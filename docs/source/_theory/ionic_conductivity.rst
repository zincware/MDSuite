Ionic Conductivity
==================
Ionic conductivity describes how charge is conducted by the carriers in a system.
In the case of molecular dynamics simulations on an atomistic level, these carriers consist of the atoms in the system
with a charge assigned to them by some level of theory.

Einstein Method
---------------
The Einstein approach to calculating the ionic conductivity utilizes the mean square displacement of the translational
dipole moment written as

.. math::
    \sigma_{E} = \frac{d}{dt}\frac{1}{6 k_{B} T V} \Bigg \langle \Bigg ( \mathbf{M}(t) - \mathbf{M}(0) \Bigg )^{2} \Bigg \rangle

where :math:`\mathbf{M}(t)` is the translational dipole moment of the system, calculated by

.. math::
    \mathbf{M}(t) = q \sum_{i} z_{i}\mathbf{r}_{i}(t)

Green-Kubo Method
-----------------
In keeping with the typical relationship between the Green-Kubo (GK) and the Einstein methods, the GK approach takes the
autocorrelation with the respect to the ionic current in the system :math:`\mathbf{J}` defined by

.. math::
    \mathbf{J}(t) = q \sum_{i} z_{i}\mathbf{v}_{i}(t).

The full equation for the conductivity is

 .. math::
    \sigma_{GK} = \frac{V}{k_{B} T} \int_{0}^{\infty} dt \langle \mathbf{J}(t) \cdot \mathbf{J}(0) \rangle

Diffusion Summation
-------------------
As an alternative to using a direct calculation, a summation of diffusion coefficients may be used to calculate the
ionic conductivity as

.. math::

    \sigma_{DS} = \rho q^{2} \frac{1}{k_{B} T} \Bigg ( x_{\alpha} z^{2}_{\alpha} D_{\alpha} +
                                                        x_{\beta} z^{2}_{\beta} D_{\beta} +
                                                        x^{2}_{\alpha} z^{2}_{\alpha} D_{\alpha \alpha} +
                                                        x^{2}_{\beta} z_{\beta}^{2} D_{\beta \beta} +
                                                        2 x_{\alpha} x_{\beta} z_{\alpha} z_{\beta} D_{\alpha \beta}
                                                \Bigg )

Nernst-Einstein Equation
------------------------
In the case of the above equation, we have included diffusion terms associated with correlated ion motion.
In the case of a system wherein distinct atoms do not interact, the above equation reduces to what is knowns as the
Nernst-Einstein equation of ionic conductivity as

.. math::

    \sigma_{DS} = \rho q^{2} \frac{1}{k_{B} T} \Bigg ( x_{\alpha} z^{2}_{\alpha} D_{\alpha} +
                                                        x_{\beta} z^{2}_{\beta} D_{\beta}
                                                \Bigg )

The Nernst-Einstein equation can be used to determine how the ions within a system interact with each other, and how
this interaction impacts the dynamics of the material.
