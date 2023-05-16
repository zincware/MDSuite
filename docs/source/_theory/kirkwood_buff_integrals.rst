Kirkwood-Buff Integrals
=======================
The Kirkwood-Buff integrals give users information about thermodynamic properties of a system, and are directly related
to the structure found from the radial distribution function.
This makes their calculation very interesting for scientists, as it allows for relevant information regarding the
thermodynamics or energetics of the system to be extracted simply by studying the structure of the particles.
The Kirkwood-Buff integrals are related to the RDF by

.. math::

    G_{\alpha \beta}(r) = \int_{0}^{\infty} dr (g_{\alpha \beta}(r) - 1)

Implementation
--------------
In MDSuite we simply perform this integral and save a function corresponding to the correctly normalized Kirkwood-Buff
integrals.
