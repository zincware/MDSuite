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
Perform a functional test on two molten salts.
"""
from typing import Tuple

import pytest
from zinchub import DataHub

import mdsuite as mds


@pytest.fixture(scope="session")
def traj_files(tmp_path_factory) -> Tuple[str, str]:
    """Download trajectory file into a temporary directory and keep it for all tests"""
    temporary_path = tmp_path_factory.getbasetemp()

    NaCl_file = DataHub(url="https://github.com/zincware/DataHub/tree/main/NaCl_rnd_md")
    NaCl_file.get_file(temporary_path)
    NaCl_out = (temporary_path / NaCl_file.file_raw).as_posix()

    KCl_file = DataHub(url="https://github.com/zincware/DataHub/tree/main/KCl_rnd_md")
    KCl_file.get_file(temporary_path)
    KCl_out = (temporary_path / KCl_file.get_file(temporary_path)).as_posix()

    return NaCl_out, KCl_out


def mdsuite_project(traj_files, tmp_path) -> mds.Project:
    """
    Create the MDSuite project and add data to be used for the rest of the tests.

    Parameters
    ----------
    traj_files : tuple
            Files include:
                * NaCl Simulation
                * KCl Simulation
    tmp_path : Path
            Temporary path that may be changed into.

    Returns
    -------
    project: mdsuite.Project
            An MDSuite project to be tested.
    """
    project = mds.Project(storage_path=tmp_path.as_posix())

    na_cl_file, k_cl_file = traj_files

    project.add_experiment(
        name="NaCl",
        timestep=0.002,
        temperature=1200.0,
        units="metal",
        simulation_data=na_cl_file,
    )
    project.add_experiment(
        name="KCl",
        timestep=0.002,
        temperature=1200.0,
        units="metal",
        simulation_data=k_cl_file,
    )

    return project


def test_analysis():
    """
    Perform analysis on these MD simulations and ensure the outcomes are as expected.

    Returns
    -------
    Test the following:

    * Two experiments added to a project successfully

    Notes
    -----
    TODO: Add correct tests when all post-RDF calculators are fixed.
    """
    pass
