Memory Management
=================
Memory management is performed under two separate operations in MDSuite, the addition of data to the database, and the
calculation of properties from the trajectory.

Definitions
^^^^^^^^^^^
 - **memory usage**: amount of memory used by a single configuration of a single property
 - **batch size**: number of configurations of size **memory usage** that can fit into memory at one time
 - **number of batches**: Number of batches that can be loaded from the total trajectory

Database Construction
^^^^^^^^^^^^^^^^^^^^^
In the initial stages of using MDSuite a database will be constructed. Depending on the size of the trajectory being
analyzed, this will require memory management, as all of the data cannot be loaded into memory at one time. In these
cases, MDSuite performs trivial memory management by only reading in N configurations at a time, where N is calculated
in a pre-processing stage. This occurs as follows:

1. Determine the number of configurations and atoms in the file.
2. Calculate the amount of memory per configuration.
3. Determine how many batches of N configurations can be loaded from the file
4. Loop over the number of batches and read in N configurations in each one.

Calculators
^^^^^^^^^^^
In the case of calculations, MDSuite employs a form of asymmetric memory management in order to maximise the amount
of data that can be used from the trajectory. This all calculated at the run time of the calculators in the following
steps:

1. Collect memory information for all datasets in hdf5 database (pre-stored in the experiment class).
2. Calculate the amount of configurations of a single property that can fit into memory at one time. If this value is
   greater than the total number of configurations, the batch size is set to the number of configurations. If not, the
   maximal batch size is set at this number.
3. Determine how many of these batch sizes can be loaded from the trajectory and balance the final batch loads so that
   unnecessary amounts of data are not lost.
4. Loop over the number of batches, load a batch, and perform the calculation on all of the data as usual.

Some of the MDSuite calculators expand the data by different amounts during calculation. In these cases, it is
important that the batch size reflects the fact that more than just the configuration will be required. In these cases,
a scaling factor is applied to the memory properties to reflect that some memory must be reserved for the calculation
that will take place. This is particularly evident in the case of Einstein calculations and RDF calculations which
replicate frames in order to perform vectorized operations.

Parallelization
^^^^^^^^^^^^^^^
In the case of a parallel run, it is important to ensure that memory limits are still being respected. Currently,
MDSuite implements a naive calculation where it is assumed that parallelization will occur over species calculations.
In these cases, the batch size is modified as if all species was going to be loaded at any one time. For a symmetric
system, this would be half the size of the serialized calculation. This is however, not always necessary. In the case
of a parallel calculation over ensembles, there would be no need to change the memory by such a factor, merely by the
scaling factor to reflect that this operation would be occurring simultaneously.

Future Development
^^^^^^^^^^^^^^^^^^
In the future, MDSuite will implement a type of memory/parallelization optimization scheme wherein the optimal amount
of parallelization and data loading is performed to maximise performance.
