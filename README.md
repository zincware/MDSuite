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
git clone https://github.com/SamTov/MDSuite.git
cd MDSuite
pip install .
```
For an overview of the program functionality, run the module directly with `-h` as an input flag.
```
python -m mdsuite -h
```

## Data Format
This post-processing suite will take a LAMMPS file and will determine what properties are available for use in the 
analysis. It also supports coordinate unwrapping as well as trivial calculations in order to minimize data requirements.
Currently, the only expectation from the lammps dump is that the group names are in the third column. This is a bug 
and will be corrected in the near future.

## Analysis 
Currently the program can perform the following analysis.

### Self Diffusion Coefficients
Self diffusion coefficients describe the average rate at which atoms move through the bulk material and can provide 
useful insights into ionic interactions taking place within a material. In this program, there are two methods by which 
we calculate the diffusion coefficients.
#### Einstein Relation
One of the methods we implement here is the Einstein method of diffusion coefficient calculation. This method involves 
calculating the mean square displacement of the positions of atoms through time and averaging over atoms as well as 
reference times. 
#### Green-Kubo Diffusion Coefficients
Another method used for the calculation of many dynamic quantities is the Green-Kubo relation for diffusion. In this 
model, we calculate the velocity autocorrelation function for each atom, averaging over times and atoms, in order to 
calculate the diffusion coefficients. The Green-Kubo implementation has the added benefit of sampling over different
trajectory ranges, thus allowing for a reasonable estimate of the error in the calculation. This will be adjusted in the
future to allow for sampling over the data correlation time, thereby giving a better approximation for the error.

### Ionic Conductivity
Another useful property of many liquid systems is ionic conductivity through the bulk. By studying the ionic 
conductivity one can learn not only about possible applications but also the ionic effect present in the system.
Currently we offer two methods for the calculation of the ionic conductivity.
#### Einstein-Helfand Relationships
In the Einstein-Helfan relations, the mean-square-displacement of the system dipole is calculated and plotted against
time. From this plot, the linear region is fit using a fitting function, and the gradient taken to be the ionic
conductivity.

#### Green-Kubo Ionic Conductivity
This is an implementation of another Green-Kubo relation, wherein the current autocorrelation function is calculated 
and integrated over, in order to calculate the ionic conductivity.

## Tools
### Trajectory Unwrapper
Often when performing simulations it is helpful to store wrapped coordinates in order to calculate structures and 
better understand the PBC effects. In order to calculate dynamics using MSD approaches, this must be undone. Therefore, 
we have written a coordinate unwrapping method inside the Trajectory class which may be used to automatically accomplish
this task.

### XYZ writer
In ths package we have also written a tool to print an xyz file from the data. This will allow you to visualize the 
unwrapped coordinates for example.

## Future Releases
In the near future we hope to provide functionality for the following properties. 

* Radial Distribution Functions
* Coordination Numbers
* Structure Factor
* Viscosity
* Phonon Spectrum

