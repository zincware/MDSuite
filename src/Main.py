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

    NaCl_1300K = Trajectory(analysis_name = "1300K_VACF",
                            storage_path = "/data/stovey",
                            new_project = False)
    NaCl_1300K.Green_Kubo_Conductivity(100000)

    NaCl_1350K = Trajectory(analysis_name="1350K_VACF",
                            storage_path="/data/stovey",
                            new_project=False)
    NaCl_1350K.Green_Kubo_Conductivity(100000)

    NaCl_1400K = Trajectory(analysis_name="1400_VACF",
                            storage_path="/data/stovey",
                            new_project=False)
    NaCl_1400K.Green_Kubo_Conductivity(100000)

    NaCl_1450K = Trajectory(analysis_name="1450K_VACF",
                            storage_path="/data/stovey",
                            new_project=False)
    NaCl_1450K.Green_Kubo_Conductivity(100000)



if __name__ == "__main__":
    main()
