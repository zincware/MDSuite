# Important Note

Package not yet ready for release. Expected initial release September 2020. Feel free to use it as it is and leave
issues in the github repository.

## Disclaimer

Currently this project in in development. It is likely that the methods in the Trajectory class will be changed in the 
next months and so it is important to update the repository before using. 

Furthermore, we make no assurances as to the accuracy of the results. It is your own responsibility to ensure you are
comfortable with returned values before publishing.  

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
