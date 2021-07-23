"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Module to contain the structured dict for file io.

Summary
-------
"""
from mdsuite.file_io.lammps_trajectory_files import LAMMPSTrajectoryFile
from mdsuite.file_io.lammps_flux_files import LAMMPSFluxFile
from mdsuite.file_io.extxyz_trajectory_reader import EXTXYZFileReader

dict_file_io = {
    "lammps_traj": (LAMMPSTrajectoryFile, "traj"),
    "lammps_flux": (LAMMPSFluxFile, "flux"),
    "extxyz": (EXTXYZFileReader, "traj"),
}
