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

import numpy as np
import pytest
from zinchub import DataHub

import mdsuite as mds


@pytest.fixture(scope="session")
def traj_file(tmp_path_factory) -> str:
    """Download trajectory file into a temporary directory and keep it for all tests"""
    temporary_path = tmp_path_factory.getbasetemp()

    NaCl = DataHub(url="https://github.com/zincware/DataHub/tree/main/NaCl_gk_i_q")
    NaCl.get_file(path=temporary_path)

    return (temporary_path / NaCl.file_raw).as_posix()


@pytest.fixture(scope="session")
def true_values() -> dict:
    """Example fixture for downloading analysis results from github"""
    NaCl = DataHub(url="https://github.com/zincware/DataHub/tree/main/NaCl_gk_i_q")
    return NaCl.get_analysis(analysis="CoordinationNumbers.json")


def test_project(traj_files, true_values, tmp_path):
    """Test the CN called from the project class"""
    os.chdir(tmp_path)
    project = mds.Project()
    project.add_experiment(
        "NaCl", simulation_data=traj_files[0], timestep=0.002, temperature=1400
    )

    computation = project.run.CoordinationNumbers()

    keys = project.run.CoordinationNumbers.result_keys

    data_dict = computation["NaCl"].data_dict["Na_Na"]
    # TODO issue that  the CN outputs will only contain a single cn, not both
    #  fix later
    np.testing.assert_array_almost_equal(data_dict[keys[0]], true_values["x"])
    np.testing.assert_array_almost_equal(data_dict[keys[1]], true_values["y"])
    np.testing.assert_array_almost_equal(data_dict[keys[2]], true_values["uncertainty"])
