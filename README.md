# Important Note

Package not yet ready for release. Expected initialty release September 7th 2020

# LAMMPS Tools
A post-processing suite for the [LAMMPS](https://lammps.sandia.gov/) simulation package.

This program takes the LAMMPS dump file and process it to calculate several properties of the systems often of interest 
in molecular dynamics simulations. Upon finishing the desired analysis, binary files of the available data are stored in a new directory for reload purposes if further analysis is required. The calculated properties are stored and appended to a .txt file in this new directory.

## Disclaimer

Currently this project in in development. It is likely that the methods in the Trajectory class will be changed in the 
next months and so it is important to update the repository before using. 

Furthermore, we make no assurances as to the accuracy of the results. It is your own responsibility to ensure you are
comfortable with returned values before publishing.  
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

## Analysis 
This suite is capable of calculating several properties both structural and dynamic of a system. The following is a 
breakdown of the capabilities of the program.

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
calculate the diffusion coefficients.

Each of these modules will save images associated with the analysis as well as raw data files which can be 
re-visualized using the tools provided in the package
### Ionic Conductivity
Another useful property of many liquid systems is ionic conductivity through the bulk. By studying the ionic 
conductivity one can learn not only about possible applications but also the ionic effect present in the system.
#### Einstein-Helfand Relationships
#### Green-Kubo Ionic Conductivity

## Future Releases
In the near future we hope to provide functionality for the following properties. 

* Radial Distribution Functions
* Coordination Numbers
* Structure Factor
* Viscosity
* Phonon Spectrum
* XYZ writer

