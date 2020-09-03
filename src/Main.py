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

    NaCl_1074K = Trajectory(analysis_name="1074K_VACF",
                      storage_path="/data/stovey",
                      new_project=False,
                      temperature=1074.0,
                      time_step=0.002,
                      time_unit=1e-12,
                      filename="/beegfs/work/stovey/LAMMPSSims/NaCl/scaledSim/Thesis_Sims/1074K/VACF/NaCl_Velocities.xyz")

    NaCl_1148K = Trajectory(analysis_name="1148K_VACF",
                      storage_path="/data/stovey",
                      new_project=True,
                      temperature=1148.0,
                      time_step=0.002,
                      time_unit=1e-12,
                      filename="/beegfs/work/stovey/LAMMPSSims/NaCl/scaledSim/Thesis_Sims/1148K/VACF/NaCl_Velocities.xyz")

    NaCl_1174K = Trajectory(analysis_name="1174K_VACF",
                      storage_path="/data/stovey",
                      new_project=True,
                      temperature=1174.0,
                      time_step=0.002,
                      time_unit=1e-12,
                      filename="/beegfs/work/stovey/LAMMPSSims/NaCl/scaledSim/Thesis_Sims/1174K/VACF/NaCl_Velocities.xyz")

if __name__ == "__main__":
    main()
