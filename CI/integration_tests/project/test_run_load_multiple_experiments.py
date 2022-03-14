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

import os

import pytest
from zinchub import DataHub

import mdsuite as mds


@pytest.fixture(scope="session")
def traj_files(tmp_path_factory) -> list:
    """Download files into a temporary directory and keep them for all tests"""
    base_url = "https://github.com/zincware/DataHub/tree/main"

    files_to_load = [
        "NaCl_gk_i_q.lammpstraj",
        "NaCl_gk_ni_nq.lammpstraj",
        "NaCl_i_q.lammpstraj",
        "NaCl_ni_nq.lammpstraj",
    ]

    temporary_path = tmp_path_factory.getbasetemp()
    file_paths = []
    for fname in files_to_load:
        folder = fname.split(".")[0]
        url = f"{base_url}/{folder}"
        dhub_file = DataHub(url=url, tag="v0.1.0")
        dhub_file.get_file(temporary_path)
        file_paths.append((temporary_path / dhub_file.file_raw).as_posix())

    return file_paths


@pytest.fixture(scope="session")
def project(tmp_path_factory) -> mds.Project:
    """Create a project in a temporary directory"""
    os.chdir(tmp_path_factory.getbasetemp())
    project = mds.Project()

    return project


def test_add_run_load_data(project, traj_files):
    """Add data and run RDF on all of them

    Test the run_computation and load_data method on multiple experiments
    """
    project.add_experiment(
        "NaCl0", simulation_data=traj_files[0], timestep=0.002, temperature=1400
    )
    project.add_experiment(
        "NaCl1", simulation_data=traj_files[1], timestep=0.002, temperature=1400
    )
    project.add_experiment(
        "NaCl2", simulation_data=traj_files[2], timestep=0.002, temperature=1400
    )
    project.add_experiment(
        "NaCl3", simulation_data=traj_files[3], timestep=0.002, temperature=1400
    )

    # Check that 4 experiments have been created
    assert len(project.experiments) == 4

    loaded_data = project.run.RadialDistributionFunction(plot=False)

    # Check that data for 4 experiments has been loaded
    assert len(loaded_data) == 4
    # Check the keys
    assert {x for x in loaded_data} == {"NaCl0", "NaCl1", "NaCl2", "NaCl3"}  #

    # Each loaded data should contain 3 entries, Na-Na, Na-Cl, Cl-Cl
    for data in loaded_data.values():
        assert len(data.data_dict) == 3
