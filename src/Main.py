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

    NaCl_1400K = Trajectory("NaCl_Melt_1300K")
    NaCl_1400K.Build_Database()
    
    # NaCl_1400K.Unwrap_Coordinates()
    # NaCl_1400K.Einstein_Diffusion_Coefficients()
    # NaCl_1400K.Green_Kubo_Diffusion_Coefficients()
    # NaCl_1400K.Green_Kubo_Conductivity()

    # Unwrap test
    #unwrap_test = Trajectory("Unwrap_Test")
    #unwrap_test.Build_Database()
    #unwrap_test.Unwrap_Coordinates()

    ########################################################
    #                   True Code Structure                #
    ########################################################
    # Begin_Program()  # Run the first initialization with input flags


if __name__ == "__main__":
    main()

