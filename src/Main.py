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
    test = Trajectory(analysis_name="test_Analysis",
                      storage_path="/tikhome/stovey/work/Repositories/MDSuite/tests",
                      new_project=False,
                      temperature=1400.0,
                      time_step=0.002,
                      time_unit=1e-12,
                      filename="/tikhome/stovey/work/Repositories/MDSuite/tests/LiF_sample.xyz")

    #NaCl_1300K.Green_Kubo_Conductivity(1250, plot=True)
    test.Green_Kubo_Diffusion_Coefficients()

if __name__ == "__main__":
    main()
