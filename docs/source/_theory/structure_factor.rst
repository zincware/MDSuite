Structure Factor
================

The structure factor describes how incident radiation scatters from material.The scattering of X-ray radiation can be
described using the partial structure factors :math:`S_{\alpha \beta}`, which were
introduced by `Faber and Zimann <https://www.tandfonline.com/doi/abs/10.1080/14786436508211931>`_.
Using this formalism the partial structure factor in dependence of the magnitude of the scattering vector :math:`Q`

.. math::
    S_{\alpha \beta}(Q) = 1 + 4\pi \rho \int _{0}^{\infty} \mathrm{d}r r^2 \frac{\sin{Qr}}{Qr} (g_{\alpha \beta}(r)-1).

can be calculated requiring only knowledge of the radial distribution function :math:`g_{\alpha \beta }(r)` and the particle density :math:`\rho`.
The total structure factor :math:`S(Q)` is calculated using a weighted sum of the partial structure factors
as described in `Tovey et. al. <https://pubs.acs.org/doi/10.1021/acs.jpcc.0c08870>`_ and
`Keen <http://scripts.iucr.org/cgi-bin/paper?S0021889800019993>`_

.. math::
    S(Q) = \sum _{\alpha \beta} w_{\alpha \beta}^{x} S_{\alpha \beta} (Q).

The weights are determined using the atomic form factors :math:`f_i(Q)` of species i and the molar fraction :math:`c_i`

.. math::
    w_{\alpha \beta}^{x} = c_{\alpha} c_{\beta} \frac{f_{\alpha}(Q)f_{\beta}(Q)}{\sum _{i=1}^{n} c_i f_i(Q)}.

The atomic form factors depend on the type and charge of the element and are approximated by a sum of gaussians, whose
coeffients are taken from `TU Graz <http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php>`_.
The atomic form factors are valid for a range of :math:`0 < Q < 25 \, \AA ^{-1}`.
In order for the calculation to work the element names and charges need to be set in the species dictionary
as shown in the sample script structure_factor.py .

