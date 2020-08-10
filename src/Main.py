"""
Author: Samuel Tovey
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Purpose: Main functionality of the LAMMPS analysis suite
"""

# 1300K: 15863.2
# 1450K: 19151.0
from sys import getsizeof
from Routines import *
from Classes import *
from UI import *


def main():
    """ Main function to coordinate use of the program """

    trajectory_class = Begin_Program()  # Run the first initialization with input flags

    # Select_Analysis() # Choose the analysis to be run
    # Run_Analysis() # Run the desired analysis
    # Write_Summary # Write a summary of the analysis

    #trajectory_class.Unwrap_Coordinates()
    #trajectory_class.Einstein_Diffusion_Coefficients()
    #trajectory_class.Green_Kubo_Diffusion_Coefficients()
    #trajectory_class.Green_Kubo_Conductivity()
    trajectory_class.Einstein_Helfand_Conductivity()




if __name__ == "__main__":
    main()
