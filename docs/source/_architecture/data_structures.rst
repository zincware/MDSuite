Data structures
===============

One of the biggest selling points of the MDSuite program is our use the database structures in the post-processing
of simulations. We use four different methods of data storage in the MDSuite code, all of which have been chosen
to meet specific needs. We will discuss all of these structure below, but include short summaries here for reference:

- **Simulation data**: This is data taken from the simulation files including trajectories, temperatures, system
  pressure and other information related to an actual simulation. This data is stored in a hdf5 database using
  scale offset compression.
- **Calculation data**: This is data that is generated during a calculation such as an autocorrelation function or an
  MSD plot. This data is stored separately to the main simulation data so as to be easily moved around and accessed
  by users independently. A hdf5 database is also used for these datasets as it allows for nice structuring of the
  information. However, in this database the data is compressed using the gzip lossless method.
- **Calculation results**: Calculation results are the scalar values calculated from an analysis such as diffusion
  coefficients, ionic conductivity, or viscosity to name a few. This data is stored in a YAML configuration file so
  that it is both easily accessible by the code, and easily readable as a human. This file contains all of the system
  information that is collected during analysis as well such as temperature, pressure, and other scalar values that
  define a system.
- **Class state information**: Class state information is solely used by the Experiment and Project classes. These are
  pickled files (binary data) that is written in such a way that it is very easy to load a class state and to save one
  as more information is added. The information in these files is not intended for users to read as everything they
  need should be stored in the other data structure.

HDF5 Database
^^^^^^^^^^^^^
Before we discuss the details of each data structure, we will present a brief review of the HDF5 database.
As per the groups website,

               " The HDF Group is a non-profit organization with the mission of advancing state-of-the-art open source
               data management technologies, ensuring long-term access to the data, and supporting our dedicated and
               diverse user community.The HDF Group is the developer of HDF5®, a high-performance software library and
               data format that has been adopted across multiple industries and is the de factor standard in the
               scientific and research community."

Despite its standardization accords the research community, it has note yet made much of an impact in the field of
molecular dynamics simulations. The HDF here is an acronym for Hierarchical data structure, which describes exactly
what it is, and how it works. Data inside a HDF5 database is broken into groups, subgroups, and then datasets. This
allows for the simple splitting of data into relevant clusters for later use. Furthermore, the H5py library
(pythons implementation of HDF5 databases), allows for trivial data compression, thereby reducing also storage
requirements.

For more information regarding this technology, see the `HDF5 <https://www.hdfgroup.org/>`_ or
`H5py <https://www.h5py.org/>`_ websites directly.

Simulation Data
^^^^^^^^^^^^^^^
The simulation data database, which in MDSuite is named database.hdf5, contains all time-based information from a
simulation. This includes but is not limited to:

- Positions
- Velocities
- Unwrapped positions
- Global Pressures
- Stress tensors (global or per atom)
- Temperature
- Box size
- Forces
- Ionic Current

and so on. Essentially, anything that can be measured through time in a simulation, can be stored in the simulation
database. What is not stored here, is global scalar values which are set once and never change. This includes a set
temperature, density, or calculated properties such as diffusion coefficients and so on.
**Now we need a picture**

Properties Data
^^^^^^^^^^^^^^^^
During the calculation of properties often a new function is constructed from which properties can be determined. This
could be a correlation function, and distribution function, or simply some series data over coordination numbers. From
this data we will often extract some single value as a result such as the diffusion coefficients or a single shell
coordination number. These values will depend not only on the system being studied, but also several parameters that
were used during the calculation. It is important to store and record this data as it allows for accurate, persistent
records which are essential in a publication. To this end, MDSuite stores all data from a calculation into an SQL
database which we have named the Properties database. This data is accompanied by the parameters of a calculation such
as data range, number of configurations, and even correlation time.

Class State Information
^^^^^^^^^^^^^^^^^^^^^^^
The final data structure used by MDSuite is the class state file. These files are generated for both the Project class
and the Experiment class and hold information about the "state" or current attributes of the class. These files are
saved using the python pickle functionality and are stored under the file extension .bin. They are not intended to be
read by the users and are only used by the classes to which they belong.
