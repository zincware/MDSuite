:orphan:
HDF5 Database Structure
=======================

One of the most unique parts of MDSuite is the usage of a hdf5 database for the organization and storing of simulation
trajectories. For this very reason, it is important that we have an informative resource for uses so that they can
understand not only why we use hdf5, but also what it is and how it works.

Hierarchical Database Structures
--------------------------------
HDF5 is an acronym for Hierarchical Data Format 5 and was developed initially by National Center for Super Computing
Applications in the 1980's and 90's. History aside, the interesting, and in our cases, most useful aspect of HDF5 is
how it stores data.

In a HDF5 database there are two objects, groups, and datasets. Groups act as containers for datasets to be added to
and can be further expanded into subgroups, something we take great advantage of. The datasets themselves are
multidimensional arrays of an arbitrary type, although depending on the type, the method of compression will become
important.

How does HDF5 Access Data?
--------------------------
Access to elements of a HDF5 database, or indexing as it is called, is performed using what is known as a B-tree method.
The B-tree search algorithm allows for access to data in logarithmic time, outperforming the standard SQL approach
to data indexing. This algorithm is itself hierarchical, as it keeps a record of data in a tree like structure.

Data Compression in MDSuite
---------------------------
There are many ways to compress data in a stored structure, and depending on what type of data you are storing, it pays
to pay attention to which one you are using. In the case of MD simulations, we are storing coordinates for the most part
, and almost always a floating point number. For this reason, we have deemed it okay to compress data in a lossy
way, which is to say, by the loss of information in the stored data.

Specifically, we use what is know as a Scale Offset filter to compress the simulation data in the HDF5 datasets. The
method works by offsetting the data by a minimum in the array and then scaling it by a factor decided by the user. This
often leads to the storing of integer data rather than the original floats. This results in better compression than
many of the current methods, and a very limited loss in accuracy.