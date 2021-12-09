"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description:
"""
import os

import pytest
from zinchub import DataHub

import mdsuite
import mdsuite as mds
import mdsuite.transformations
from mdsuite.utils.testing import assertDeepAlmostEqual


@pytest.fixture(scope="session")
def traj_file(tmp_path_factory) -> str:
    """Download trajectory file into a temporary directory and keep it for all tests"""
    temporary_path = tmp_path_factory.getbasetemp()

    NaCl = DataHub(url="https://github.com/zincware/DataHub/tree/main/NaCl_gk_i_q")
    NaCl.get_file(path=temporary_path)

    return (temporary_path / NaCl.file_raw).as_posix()


@pytest.fixture()
def mdsuite_project(traj_file, tmp_path) -> mdsuite.Project:
    project = mdsuite.Project(storage_path=tmp_path.as_posix())
    project.add_experiment("NaCl", simulation_data=traj_file)

    return project


def test_from_project(mdsuite_project):
    mdsuite_project.run.CoordinateUnwrapper()


def test_from_project_twice(mdsuite_project):
    mdsuite_project.run.CoordinateUnwrapper()
    #mdsuite_project.run.CoordinateUnwrapper()


def test_from_experiment(mdsuite_project):
    mdsuite_project.experiments.NaCl.run.CoordinateUnwrapper()


def test_from_experiment_twice(mdsuite_project):
    mdsuite_project.experiments.NaCl.run.CoordinateUnwrapper()
    mdsuite_project.experiments.NaCl.run.CoordinateUnwrapper()


def test_pass_instance_to_exp(mdsuite_project):
    mdsuite_project.experiments.NaCl.cls_transformation_run(
        mdsuite.transformations.CoordinateUnwrapper()
    )
