# LAMMPS Tools
A post-processing suite for the [LAMMPS](https://lammps.sandia.gov/) simulation package.

This program takes the LAMMPS dump file and process it to calculate several properties of the systems often of interest in molecular dynamics simulations. Upon finishing the desired analysis, binary files of the available data are stored in a new directory for reload purposes if further analysis is required. The calculated properties are stored and appended to a .txt file in this new directory.

## Installation
Clone the repository and run the main.py script.
```
python main.py
```
This will bring up lists of analysis which may be performed on the system.
## Data Format
This post-processing suite will take a LAMMPS file and will determine what properties are available for use in the analysis. It also support coordinate unwrapping as well as trivial calculations in order to minimize data requirements.

## Requirements
Requirements.txt contains a full list of python packages required to use this suite but another list with links is given here.
* [Numpy](https://numpy.org/)
* [Scipy](https://www.scipy.org/)
* [MDAnalysis](https://www.mdanalysis.org/)

## Analysis 
This suite is capable of calculating several properties both structural and dynamic of a system. The following is a breakdown of the capabilities of the program.

### Radial Distribution Functions
Radial distribution functions are a good starting point for understanding the structure of a material. This program uses the MDAnalysis software to calculate the RDF's for each element and element combination within the system. It will produce .npy files containing the raw data for the RDF's as well as the plots. This raw data can be re-plotted using some of the post-analysis tools in the tools directory. 
[rdfimage](Images/)

### Coordination Numbers
Coordination numbers provides additional data on the neighbor density of the atoms within a system. This allows one in particular to study effects of structure reduction under varying conditions within a system.
[cnimages](Images/)

### Self Diffusion Coefficients
Self diffusion coefficients describe the average rate at which atoms move through the bulk material and can provide useful insights into ionic interactions taking place within a material. In this program, there are two methods by which we calculate the diffusion coefficients.
#### Einstein Relation
One of the methods we implement here is the Einstein method of diffusion coefficient calculation. This method involves calculating the mean square displacement of the positions of atoms through time and averaging over atoms as well as reference times. 
#### Green-Kubo Diffusion Coefficients
Another method used for the calculation of many dynamic quantities is the Green-Kubo relation for diffusion. In this model, we calculate the velocity autocorrelation function for each atom, averaging over times and atoms, in order to calculate the diffusion coefficients.

Each of these modules will save images associated with the analysis as well as raw data files which can be re-visualized using the tools provided in the package
### Ionic Conductivity
Another useful property of many liquid systems is ionic conductivity through the bulk. By studying the ionic conductivity one can learn not only about possible applications but also the ionic effect present in the system.
#### Einstein-Helfand Relationships
#### Green-Kubo Ionic Conductivity

## Future Releases

### Structure Factor

### Viscosity

### Phonon Spectrum

### XYZ writer

