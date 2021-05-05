Experimental Calculators
========================

MDSuite has a number calculators, many of which are optimized, memory safe, within our coding guidelines, and tested
against publications for accuracy. Several however, fail to meet our standards in one or more of the above categories.
We call these calculators experimental and make them available at the minimum, the computation that they say
they will run will be run. In this section we outline briefly which calculators have this label and give some reasons
as to why and what to look out for when you are using them.

Radial Distribution Functions
-----------------------------
The radial distribution function only has the experimental title due to coding standards. We use a specific parent class
structure for our calculators and whilst the RDF calculator is highly optimized and accurate, it does not meet these
standards. We are working to bring it into a more uniform structure.

Angular Distribution Functions
------------------------------
In the case of the Angular distribution functions the same comments as for the RDF apply but with some additional
points.

    * Use of `tf_function=True` can result in memory overload and we do not have a means to prevent this.
    * The normalization factor is not yet added. This mean bonds at any distance up to the cutoff are treated equally.

Distinct Diffusion Coefficients
-------------------------------
In the case of the distinct diffusion coefficients, both Green-Kubo and Einstein, we have simply not checked it against
publication data and have thus far only used it to highlight that one finds a certain type of distinct correlation
function when studying subsets of systems.
