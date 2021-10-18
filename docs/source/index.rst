MDSuite documentation
=====================

Docs currently out of date.

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
   tools

.. toctree::
   :maxdepth: 1
   :caption: Technical Stuff:

   modules_and_classes
   bibliography
   developer

What is MDSuite?
----------------
MDSuite is a python package designed for efficient, powerful, and modern post-processing of molecular dynamics
simulations. However, it is not simply a tool for analysis, it is a full environment to aid in better understanding
your simulation data. This all starts with the database structure of MDSuite.

**Simulation Database**

When an experiment is added to MDSuite
the simulation data is loaded into a hdf5 database, this not only allows us to store data in a compressed fashion, but
also for faster loading and processing of trajectory data when a calculator is called.

**Analysis Database**

Of course, analysis of data is only half the battle, you also want to store the analysis and information about it so
that you can recall it and study it later. To this end, we use an SQL database structure to store not only series data
such a correlation function or distribution function, but also the physical values you extract this data such as a
diffusion coefficient or coordination numbers. This data is also stored with all of its parameters such as a
number of configurations, a data range, and even the chosen correlation time. With this storage approach, MDSuite
becomes a complete platform for analysis and studies of simulations.

Why MDSuite?
------------
With all the post-processing codes out there why should you use MDSuite? Aside
from the easy to use interface and large range of analysis available, MDSuite has
a number of great features beneath the surface that give us an edge.

* Memory safety on all calculators up to atom-wise batching.
* Built on top of TensorFlow allowing for full parallelization of processes as well as gpu use.
* 19 calculators and 11 transformations for full characterization of a simulation.
* Powerful data visualization capabilities.

On top of this, we also have a group of friendly developers who are trying the better their own skills as programmers
while sharing their knowledge of physics.

What's New?
-----------

We will keep updating this section to include new features of the code.

* Molecule mapping on raw data!
* MDSuite paper is complete.

What's Next?
------------

We have a lot going on in the direction of molecule studies as well as some more calculators specifically designed
for ab-initio calculations. We are always looking to add file reading capabilities, and of course,
improving performance. If you want to help out with anything feel free to send us an email or contact us through github.
We have included a short list below of some broad topics that we are interested in developing more aggressively in the
near future.

* Smart memory optimization
* Threaded memory adjustment
* ab-initio calculators
* binary file readers
