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
import shutil
import time
from pathlib import Path

import pytest

import mdsuite as mds
from mdsuite.database.types import MutableDict

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture()
def mdsuite_project():
    """
    Create the MDSuite project and add data to be used for the rest of the tests.
    C60 Fullerene test case to compute rings.

    Parameters
    ----------

    Returns
    -------
    project: mdsuite.Project
            An MDSuite project to be tested.
    """
    try:
        shutil.rmtree("MDSuite_Project")
        time.sleep(0.5) # this is added for windows
    except FileNotFoundError:
        pass
    traj_file = "c60.lammpstraj"
    traj_file = (Path("") / traj_file).as_posix()

    project = mds.Project()
    project.add_experiment(
        "C60", simulation_data=traj_file, timestep=0.002, temperature=1400, units="metal"
    )

    return project


def test_analysis(mdsuite_project):
    """
    Perform analysis on these MD simulations and ensure the outcomes are as expected.

    Returns
    -------
    Test the following:

        * Extraction of rings in the structure.

    """

    computation = mdsuite_project.run.FindRings(
        max_bond_length=1.48,
        number_of_configurations=2,
        plot=True,
        max_ring_size=15,
        shortcut_check=True,
    )

    assert computation['C60'].data_dict['System'] == MutableDict({'5': [12, 12], '6': [20, 20]})

