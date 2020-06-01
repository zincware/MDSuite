# LAMMPS Tools

A primitive collection of tools within a class framework which is capable of taking dump files from LAMMPS simulations and study a number of different properties. Currently this isn't functional and should not be used without serious review of the code as it is not extendable to arbitrary systems and is not optimized for speed which becomes problematic for some of the properties.

## Data Format

This code takes LAMMPS dump files of the form x y z vx vy vz fx fy fz. It will soon be extended to take arbitrary LAMMPS dump inputs

## Green Kubo

This method with calculate the Green-Kubo diffusion coefficients by calculating the integral of the velocity autocorrelation function. After the calculation of the diffusion coefficients the Nernst Einstein conductivity is calculated along with the Nernst Einstein corrections.

## Einstein-Helfand

Method to calculate the Einstein-Helfand conductivity by taking he integral over the current autocorrelation function calculated directly form velocities.
