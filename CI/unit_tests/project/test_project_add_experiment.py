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
import gzip
import os
import shutil
import urllib.request

import numpy as np
import pytest
from zinchub import DataHub
import MDAnalysis

import mdsuite as mds
import mdsuite.file_io.lammps_flux_files
import mdsuite.file_io.chemfiles_read


@pytest.fixture(scope="session")
def traj_files(tmp_path_factory) -> dict:
    """Download files into a temporary directory and keep them for all tests"""
    base_url = "https://github.com/zincware/DataHub/tree/main"

    files_to_load = [
        "NaCl_gk_i_q.lammpstraj",
        "NaCl_gk_ni_nq.lammpstraj",
        "NaCl_i_q.lammpstraj",
        "NaCl_ni_nq.lammpstraj",
        "NaCl_64_Atoms.extxyz",
        "NaClLog.log",
        "GromacsTest.gro"
    ]

    files = []
    temporary_path = tmp_path_factory.getbasetemp()
    file_paths = dict()
    for fname in files_to_load:
        folder = fname.split(".")[0]
        url = f"{base_url}/{folder}"
        print(url)
        dhub_file = DataHub(url=url)
        dhub_file.get_file(temporary_path)
        if isinstance(dhub_file.file_raw, list):
            file_paths[fname] = [(temporary_path / f).as_posix() for f in dhub_file.file_raw]
        else:
            file_paths[fname] = (temporary_path / dhub_file.file_raw).as_posix()

    return file_paths


def test_add_file_from_list(traj_files, tmp_path):
    """Check that adding files from lists does not raise an error"""
    os.chdir(tmp_path)
    project = mds.Project()
    file_names = [
        traj_files["NaCl_gk_i_q.lammpstraj"],
        traj_files["NaCl_gk_ni_nq.lammpstraj"],
    ]
    project.add_experiment(
        "NaCl", simulation_data=file_names, timestep=0.1, temperature=1600
    )

    print(project.experiments)
    assert list(project.experiments) == ["NaCl"]


def test_add_file_from_str(traj_files, tmp_path):
    """Check that adding files from str does not raise an error"""
    os.chdir(tmp_path)
    project = mds.Project()
    project.add_experiment(
        "NaCl",
        simulation_data=traj_files["NaCl_gk_i_q.lammpstraj"],
        timestep=0.1,
        temperature=1600,
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
        "NaCl",
        simulation_data=traj_files["NaCl_gk_i_q.lammpstraj"],
        timestep=0.1,
        temperature=1600,
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
        "NaCl",
        simulation_data=traj_files["NaCl_64_Atoms.extxyz"],
        timestep=0.1,
        temperature=1600,
    )
    forces = project.experiments["NaCl"].load_matrix(
        species=["Na"], property_name="Forces"
    )
    # check one value from the file
    # second timestep, Na atom nr 15
    force_shouldbe = [0.48390745, -0.99956709, 1.11229777]
    force_is = forces["Na/Forces"][15, 1, :]
    np.testing.assert_array_almost_equal(force_is, force_shouldbe, decimal=5)


def test_lammpsflux_read(traj_files, tmp_path):
    os.chdir(tmp_path)
    project = mds.Project()
    custom_headers = {
        "Time": ["Time"],
        "Temperature": ["Temp"],
        "Density": ["Density"],
        "Pressure": ["Press"],
    }
    file_reader = mds.file_io.lammps_flux_files.LAMMPSFluxFile(
        file_path=traj_files["NaClLog.log"],
        sample_rate=1000,
        box_l=3 * [42.17],
        n_header_lines=29,
        custom_data_map=custom_headers,
    )
    project.add_experiment(
        "NaCl_flux", simulation_data=file_reader, timestep=0.1, temperature=1600
    )
    pressures = project.experiments["NaCl_flux"].load_matrix(
        species=["Observables"], property_name="Pressure"
    )
    # check one value from the file
    # line 87
    press_shouldbe = -153.75652
    force_is = pressures["Observables/Pressure"][0, 57, 0].numpy()
    assert force_is == pytest.approx(press_shouldbe, rel=1e-5)


def test_gromacs_read(traj_files, tmp_path):
    os.chdir(tmp_path)
    project = mds.Project()

    topol_path, traj_path = traj_files['GromacsTest.gro']

    file_reader = mds.file_io.chemfiles_read.ChemfilesRead(
        traj_file_path=traj_path,
        topol_file_path=topol_path
    )

    project.add_experiment('xtc_test', simulation_data=file_reader)
    pos = project.experiments['xtc_test'].load_matrix(species=['C1'], property_name='Unwrapped_Positions')['C1/Unwrapped_Positions']

    # read the same file with mdanalysis and compare
    uni = MDAnalysis.Universe(topol_path, traj_path)
    C1s = uni.atoms.select_atoms('name C1')
    test_step = 42
    # advance the mdanalysis global step
    uni.trajectory[test_step]
    pos_mdsana = C1s.positions

    np.testing.assert_almost_equal(pos[:,test_step,:], pos_mdsana,decimal=5)
