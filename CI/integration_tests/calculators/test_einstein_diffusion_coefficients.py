"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Test for the einstein_diffusion_coefficients
"""

import json
import os

import pytest

import numpy as np
import urllib.request
import gzip
import shutil
from pathlib import Path

import data as static_data
import mdsuite as mds


@pytest.fixture(scope="session")
def traj_files(tmp_path_factory) -> list:
    """Download files into a temporary directory and keep them for all tests"""
    time_step = 0.002
    temperature = 1400.0
    base_url = "https://github.com/zincware/ExampleData/raw/main/"

    files_in_url = [
        "NaCl_gk_i_q.lammpstraj",
        # "NaCl_gk_ni_nq.lammpstraj",
        # "NaCl_i_q.lammpstraj",
        # "NaCl_ni_nq.lammpstraj",
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


@pytest.fixture(scope="session")
def true_values() -> dict:
    """Values to compare to"""
    static_path = Path(static_data.__file__).parent
    data = static_path / 'einstein_diffusion_coefficients.json'
    return json.loads(data.read_bytes())


def test_project(traj_files, true_values, tmp_path):
    """Test the EinsteinDiffusionCoefficients called from the project class"""
    os.chdir(tmp_path)
    project = mds.Project()
    project.add_experiment("NaCl", data=traj_files[0], timestep=0.002, temperature=1400)

    computation = project.run.EinsteinDiffusionCoefficients(data_range=300, correlation_time=1)

    keys = project.run.EinsteinDiffusionCoefficients.result_keys

    data_dict = computation["NaCl"].data_dict["Na"]

    np.testing.assert_array_almost_equal(data_dict[keys[0]], true_values['x'])
    np.testing.assert_array_almost_equal(data_dict[keys[1]], true_values['uncertainty'])


