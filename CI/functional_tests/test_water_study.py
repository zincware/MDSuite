"""
MDSuite: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
Functional test for the analysis of a GROMACS water simulation.
"""
import os

import mdsuite as mds
from mdsuite.utils import Units
import mdsuite.file_io.chemfiles_read


def test_water_analysis(tmp_path):
    """Test the EinsteinDiffusionCoefficients called from the project class"""
    os.chdir(tmp_path)
    base_dir = "/Users/samueltovey/BI_Work/14_molecules"

    gmx_units = Units(
        time=1e-12,
        length=1e-10,
        energy=1.6022e-19,
        NkTV2p=1.6021765e6,
        boltzmann=8.617343e-5,
        temperature=1,
        pressure=100000,
    )

    project = mds.Project("BI_Water_Concentration")

    traj_path = f"{base_dir}/traj_comp.xtc"
    topol_path = f"{base_dir}/run14.gro"
    file_reader = mdsuite.file_io.chemfiles_read.ChemfilesRead(
        traj_file_path=traj_path, topol_file_path=topol_path
    )
    project.add_experiment(
        name=f"14_Molecules",
        timestep=0.002,
        temperature=300.0,
        units=gmx_units,
        simulation_data=file_reader,
    )
    project.run.CoordinateWrapper()

    project.run.AngularDistributionFunction()
