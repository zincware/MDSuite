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
"""
import logging
from pathlib import Path

import mdsuite as mds

logging.basicConfig(level=logging.DEBUG)


def main_project(traj_file):
    """Test the neighbors called from the project class"""
    project = mds.Project()
    project.add_experiment(
        "C59", simulation_data=traj_file, timestep=0.002, temperature=1400, units="metal"
    )

    computation = project.run.FindRings(max_bond_length=1.5, number_of_configurations=2, plot=True, max_ring_size=15,
                                        shortcut_check=True)


if __name__ == '__main__':
    # shutil.rmtree("MDSuite_Project")
    test_file = "c59.lammpstraj"
    filepath = (Path("") / test_file).as_posix()
    main_project(filepath)
