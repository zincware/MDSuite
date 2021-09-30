"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Tests for adding data to an experiment
"""
import os

import pytest

import urllib.request
import gzip
import shutil

import mdsuite as mds


@pytest.fixture(scope="session")
def traj_files(tmp_path_factory) -> list:
    """Download files into a temporary directory and keep them for all tests"""
    time_step = 0.002
    temperature = 1400.0
    base_url = "https://github.com/zincware/ExampleData/raw/main/"

    files_in_url = [
        "NaCl_gk_i_q.lammpstraj",
        "NaCl_gk_ni_nq.lammpstraj",
        "NaCl_i_q.lammpstraj",
        "NaCl_ni_nq.lammpstraj",
    ]

    files = []
    temporary_path = tmp_path_factory.getbasetemp()

    for item in files_in_url:
        filename, headers = urllib.request.urlretrieve(
            f"{base_url}{item}.gz", filename=f"{temporary_path / item}.gz"
        )
        with gzip.open(filename, "rb") as f_in:
            new_file = temporary_path / item
            with open(new_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

            files.append(new_file.as_posix())

    return files


def test_add_file_from_list(traj_files, tmp_path):
    """Check that adding files from lists does not raise an error"""
    os.chdir(tmp_path)
    project = mds.Project()
    project.add_experiment("NaCl", data=traj_files[:1], timestep=0.1, temperature=1600)

    print(project.experiments)
    assert list(project.experiments) == ["NaCl"]


def test_add_file_from_str(traj_files, tmp_path):
    """Check that adding files from str does not raise an error"""
    os.chdir(tmp_path)
    project = mds.Project()
    project.add_experiment("NaCl", data=traj_files[0], timestep=0.1, temperature=1600)

    print(project.experiments)
    assert list(project.experiments) == ["NaCl"]


def test_add_file_from_dict(traj_files, tmp_path):
    """Check that adding files from dicts does not raise an error"""
    os.chdir(tmp_path)

    data = {"file": traj_files[0], "format": "lammps_traj"}

    project = mds.Project()
    project.add_experiment("NaCl", data=data, timestep=0.1, temperature=1600)

    print(project.experiments)
    assert list(project.experiments) == ["NaCl"]


def test_multiple_experiments(tmp_path):
    """Test the paths within the experiment classes

    Parameters
    ----------
    tmp_path:
        default pytest fixture

    """
    os.chdir(tmp_path)

    project = mds.Project()

    project.add_experiment("Test01")
    project.add_experiment("Test02")

    project_loaded = mds.Project()

    assert project.experiments.Test01.experiment_path == project_loaded.experiments.Test01.experiment_path
    assert project.experiments.Test02.experiment_path == project_loaded.experiments.Test02.experiment_path
