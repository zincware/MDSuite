:orphan:
Coordination Numbers
====================

Coordination numbers are used to describe the number of neighbours surrounding a reference atom within a system.
Such a property provides information on atomic structure, particularly in multi-component systems.
Coordination numbers are also used to understand how tht structure of systems changes with changing conditions.
During a change of state from solid to liquid, one will find a relaxation of the coordination number to finite values,
signifying the change of state.

Algorithm
---------
In MDSuite, we utilize the relationship of the coordination numbers with the radial distribution function define by

.. math::

    CN_{\alpha \beta} = 4 \pi \int_{r_{0}}^{r_{1}} dr r^{2} \rho g_{\alpha \beta}(r),

where :math:`\rho` is the number density of the :math:`\alpha \beta` combination.

Implementation
--------------
In order to calculate the coordination numbers (CN) in MDSuite the radial distribution functions (RDF) must have been
calculated previously.
The code applies the trapezoidal rule to integrate the RDF cumulatively and stores the CN function.
Following the calculation of the CN function, the radii at which the coordination number exists should be determined.
This is done by determining first the peaks in the RDF after the application of SavGol smoothing filter.
Following this, a Golden-Section search is applied between peaks in order to determine minimums in the RDF, and from
these values, the coordination numbers in successive shells.
