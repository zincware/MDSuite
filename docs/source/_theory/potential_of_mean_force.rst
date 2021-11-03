Potential of Mean-Force
=======================

Another useful structural property is the potential of mean-force (PMF).
This property provides an energetic measurement of the binding strength between atomic
species.
This property allows for a direct analysis of the changes in binding strenght with
respect to changing conditions.
The potential of mean force is related to the radial distribution function (RDF) by

.. math::

    w_{\alpha \beta}^{(2)}(r) = k_\mathrm{B} T \mathrm{ln} g_{\alpha \beta}(r)

Implementation
--------------
In MDSuite, the potential of mean force is calculated directly from the RDF.
In order to extract a scalar value from the PMF, we must find the minimum of the RDF
and determine the PMF at that location.
This is done applying a SavGol filter to smooth the RDF, before a peak finding
algorithm is applied to determine the peaks in the RDF.
These peaks are used as boundaries for a Golden-Section search which determines the
minimum of the RDF.
The radii if this minimum is then used to evaluate the PMF for the system.