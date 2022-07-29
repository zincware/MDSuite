MDSuite documentation
=====================
This package was conceived out of the idea that scientists can
perform computational experiments without filling their computers
with mislabelled and arbitrary directories. Here you will find the
documentation for the mdsuite python package, as well as significant
amount of theoretical background on the different types of analysis being
performed. If you have any questions feel free to get into contact with
us whether it is about how to use to code, some questions about theory, or
about what you might like to see in future releases, we are always happy to
hear from our users.

.. toctree::
   :maxdepth: 1
   :caption: Welcome:

   installation
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: Getting started:

   theory_introduction
   user_guide
   _architecture/data_structures

.. toctree::
   :maxdepth: 1
   :caption: Technical Stuff:

   modules_and_classes
   bibliography
   developer

What is MDSuite?
----------------
MDSuite is a python package designed for efficient, powerful, and modern post-processing of molecular dynamics
simulations.
It provides a memory safe, gpu accelerated, full parallelized environment for performing
a variety of analyses for molecular dynamics simulations.
Furthermore, MDSuite uses a project structure that allows for strict data tracking and
persistent storage of not only analysis results, but also the parameters of the analysis
for compete reproducibility.
