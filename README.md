# Introduction

# LAMMPS Tools
A post-processing suite for the [LAMMPS](https://lammps.sandia.gov/) simulation package.

MDSuite is a molecular dynamics postprocessing tool, initially built specifically for the LAMMPS simulation package.
The program will take a lammps dump file and build a HDF5 database from the raw data. The restructuring of the data
makes the immediate calculation of several system properties fast and easy. 

## Installation
Clone the repository with the following
```
$ git clone https://github.com/SamTov/MDSuite.git
$ cd MDSuite
$ pip install .
```
For an overview of the program functionality, run the module directly with `-h` as an input flag.
```
$ python -m mdsuite -h
```

NOTE: to install tensorflow with GPU support, use CUDA 10.1 and cuDNN 7.6.5. 

## Documentation

In order to see analysis walkthroughs, tool examples, software information, or contribution resources, 
please construct a local version of the documentation. This can be done as follows:
```
$ cd MDSuite/docs
$ make html
$ cd build/html
$ firefox index.html
```
Note, you can use whichever browser you want.
