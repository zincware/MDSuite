"""
Author: Samuel Tovey
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Purpose: Main functionality of the LAMMPS analysis suite
"""

from Routines import *
from Classes import *
from UI import *


def main():
    """ Main function to coordinate use of the program """

    trajectory_class, filepath = Begin_Program()  # Run the first initialization with input flags
    #trajectory_class.Unwrap_Coordinates()
    #trajectory_class.Einstein_Diffusion_Coefficients()
    #trajectory_class.Green_Kubo_Diffusion_Coefficients()
    trajectory_class.Green_Kubo_Conductivity(100000)
    #trajectory_class.Einstein_Helfand_Conductivity()
    #trajectory_class.Print_Class_Attributes()

if __name__ == "__main__":
    main()
