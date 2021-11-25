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

import urllib.request
import gzip
import shutil
import numpy as np

import mdsuite as mds


@pytest.fixture(scope="session")
def traj_files(tmp_path_factory) -> list:
    """Download files into a temporary directory and keep them for all tests"""
    # time_step = 0.002
    # temperature = 1400.0
    base_url = "https://github.com/zincware/ExampleData/raw/main/"

    files_in_url = [
        "NaCl_gk_i_q.lammpstraj",
        "NaCl_gk_ni_nq.lammpstraj",
        "NaCl_i_q.lammpstraj",
        "NaCl_ni_nq.lammpstraj",
        "NaCl_64_Atoms.extxyz",
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


# todo these need to be tested, but with the new file adding syntax
def test_add_file_from_list(traj_files, tmp_path):
    """Check that adding files from lists does not raise an error"""
    os.chdir(tmp_path)
    project = mds.Project()
    project.add_experiment(
        "NaCl", fname_or_file_processor=traj_files[:2], timestep=0.1, temperature=1600
    )

    print(project.experiments)
    assert list(project.experiments) == ["NaCl"]


def test_add_file_from_str(traj_files, tmp_path):
    """Check that adding files from str does not raise an error"""
    os.chdir(tmp_path)
    project = mds.Project()
    project.add_experiment(
        "NaCl", fname_or_file_processor=traj_files[0], timestep=0.1, temperature=1600
    )

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

    assert (
        project.experiments.Test01.experiment_path
        == project_loaded.experiments.Test01.experiment_path
    )
    assert (
        project.experiments.Test02.experiment_path
        == project_loaded.experiments.Test02.experiment_path
    )


def test_lammps_read(traj_files, tmp_path):
    os.chdir(tmp_path)
    project = mds.Project()
    project.add_experiment(
        "NaCl", fname_or_file_processor=traj_files[0], timestep=0.1, temperature=1600
    )
    vels = project.experiments["NaCl"].load_matrix(
        species=["Na"], property_name="Velocities"
    )
    # check one value from the file
    # timestep 482, Na atom id 429 (line 486776 in th file)
    vel_shouldbe = [5.2118, 6.40816, 0.988324]
    vel_is = vels["Na/Velocities"][428, 482, :]
    np.testing.assert_array_almost_equal(vel_is, vel_shouldbe, decimal=5)


def test_extxyz_read(traj_files, tmp_path):
    os.chdir(tmp_path)
    project = mds.Project()
    project.add_experiment(
        "NaCl", fname_or_file_processor=traj_files[-1], timestep=0.1, temperature=1600
    )
    forces = project.experiments["NaCl"].load_matrix(
        species=["Na"], property_name="Forces"
    )
    # check one value from the file
    # second timestep, Na atom nr 15
    force_shouldbe = [0.48390745, -0.99956709, 1.11229777]
    force_is = forces["Na/Forces"][15, 1, :]
    np.testing.assert_array_almost_equal(force_is, force_shouldbe, decimal=5)
