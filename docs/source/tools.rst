Tools
=====

As well as being a useful program for the analysis of Molecular Dynamics (MD) simulations,
MDSuite is shipped with several useful tools which can be used for various analysis. In this
section, we look into some of these tools and how they might be used.

Coordinate Unwrapper
--------------------

In order to simulate the effects of a large bulk system, periodic boundary conditions will be
applied to a simulation. This effectively states that we replicate the simulation box infinitely
in all directions such that we are only studying the bulk of the system. In order to replicate
this, when a particle moves through the side of the simulation box, we re-introduce this 
particle on the other side. Whilst this allows for accurate simulations of a bulk system, often
we want to study how particles move through a larger bulk. For this, we must unwrap the 
coordinates of the simulation box, where rather than appearing on the other side of the box, the
particles will simply continue on. 

In MDSuite, in order to calculate Einstein relations, we have implemented a coordinate Unwrapper.
This functionality is included as a method within the trajectory class. It can be called simply
by:

.. code-block:: python
        
        import mdsuite as mds
        NaCl = mds.Experiment(...)
        NaCl.unwrap_coordinates()

Upon being called, this method store the unwrapped coordinates in the HDF5 database constructed
upon instantiating the NaCl class instance.

XYZ File Writer
---------------

Another closely related tool is the XYZ file writer. It is common in an MD simulation that we 
may wish to visualize the coordinates of a system, or simply have access to only certain 
information from a larger data file. In this case, one can simply call the XYZ file writer
tool available, and print whatever data they choose. Using the configuration above, we could
print the coordinates of the unwrapped positions by:

.. code-block:: python

        NaCl.write_xyz(property='Unwrapped_Positions')

This call would save an xyz file in the working directory under the name NaCl_Unwrapped_Positions.xyz.

Transformation Classes
----------------------

.. toctree::
   :maxdepth: 1

   _usage/unwrap_coordinates


File IO
-------

.. toctree::
   :maxdepth: 1

   _usage/file_read
   _usage/lammps_trajectory_files


Utilities
---------
.. toctree::
   :maxdepth: 1

   _usage/meta_functions


Other
---------
.. toctree::
   :maxdepth: 1

   _usage/coordination_number_calculation
