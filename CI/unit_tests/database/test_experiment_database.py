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
from tempfile import TemporaryDirectory
import pytest
import mdsuite as mds
import numpy as np

temp_dir = TemporaryDirectory()
cwd = os.getcwd()


@pytest.fixture(autouse=True)
def prepare_env():
    """Prepare temporary environment"""
    temp_dir = TemporaryDirectory()
    os.chdir(temp_dir.name)

    yield

    os.chdir(cwd)
    temp_dir.cleanup()


def test_project_temperature():
    """Test that the project description is stored correctly in the database"""

    project_1 = mds.Project()
    project_1.add_experiment(experiment="Exp01")
    project_1.experiments["Exp01"].temperature = 9000

    project_2 = mds.Project()
    project_2.load_experiments("Exp01")
    assert project_2.experiments["Exp01"].temperature == 9000


def test_project_time_step():
    """Test that the project description is stored correctly in the database"""

    project_1 = mds.Project()
    project_1.add_experiment(experiment="Exp01")
    project_1.experiments["Exp01"].time_step = 1

    project_2 = mds.Project()
    project_2.load_experiments("Exp01")
    assert project_2.experiments["Exp01"].time_step == 1


def test_project_number_of_configurations():
    """Test that the project description is stored correctly in the database"""

    project_1 = mds.Project()
    project_1.add_experiment(experiment="Exp01")
    project_1.experiments["Exp01"].number_of_configurations = 100

    project_2 = mds.Project()
    project_2.load_experiments("Exp01")
    assert project_2.experiments["Exp01"].number_of_configurations == 100


def test_project_number_of_atoms():
    """Test that the project description is stored correctly in the database"""

    project_1 = mds.Project()
    project_1.add_experiment(experiment="Exp01")
    project_1.experiments["Exp01"].number_of_atoms = 100

    project_2 = mds.Project()
    project_2.load_experiments("Exp01")
    assert project_2.experiments["Exp01"].number_of_atoms == 100


def test_species():
    """Test that the species are stored correctly in the database"""

    species = {
        "H": {"indices": [1, 2, 3], "mass": 1},
        "Cl": {"indices": [4, 5, 6], "mass": 35.45},
    }

    project_1 = mds.Project()
    project_1.add_experiment(experiment="Exp01")
    project_1.experiments["Exp01"].species = species

    project_2 = mds.Project()
    project_2.load_experiments("Exp01")
    assert project_2.experiments["Exp01"].species == species


def test_molecules():
    """Test that the molecules are stored correctly in the database"""

    molecule = {
        "Proton": {"indices": [1, 2, 3], "mass": 1},
        "Chloride": {"indices": [4, 5, 6], "mass": 35.45},
    }

    project_1 = mds.Project()
    project_1.add_experiment(experiment="Exp01")
    project_1.experiments["Exp01"].molecules = molecule

    project_2 = mds.Project()
    project_2.load_experiments("Exp01")
    assert project_2.experiments["Exp01"].molecules == molecule


def test_project_box_array():
    """Test that the project description is stored correctly in the database"""

    box_array = np.array([1.0, 1.414, 1.732])

    project_1 = mds.Project()
    project_1.add_experiment(experiment="Exp01")
    project_1.experiments["Exp01"].box_array = box_array

    project_2 = mds.Project()
    project_2.load_experiments("Exp01")
    np.testing.assert_array_equal(project_2.experiments["Exp01"].box_array, box_array)


def test_experiment_simulation_data():
    """Test that the experiment simulation data is stored correctly in the database"""

    simulation_data = {
        "a_5": [10.0, 11.0, 12.0],
        "b_test": "HelloWorld",
        "c": 15.0,
    }  # can only handle float and str

    project_1 = mds.Project()
    project_1.add_experiment(experiment="Exp01")
    project_1.experiments["Exp01"].simulation_data = simulation_data

    project_2 = mds.Project()
    project_2.load_experiments("Exp01")

    for key, val in project_2.experiments["Exp01"].simulation_data.items():
        assert val == simulation_data[key]


def test_experiment_simulation_data_nested():
    """
    Test that nested experiment simulation data is stored correctly in the database
    """

    simulation_data = {"a": {"one": [1.0, 2.0, 3.0], "two": [4.0, 5.0, 6.0]}}
    simulation_true = {"a.one": [1.0, 2.0, 3.0], "a.two": [4.0, 5.0, 6.0]}

    project_1 = mds.Project()
    project_1.add_experiment(experiment="Exp01")
    project_1.experiments["Exp01"].simulation_data = simulation_data

    project_2 = mds.Project()
    project_2.load_experiments("Exp01")

    for key, val in project_2.experiments["Exp01"].simulation_data.items():
        assert val == simulation_true[key]


def test_experiment_units():
    """Test that the experiment simulation data is stored correctly in the database"""

    custom_units = {"time": 17, "length": 1e-23}

    from mdsuite.utils.units import units_real

    project_1 = mds.Project()
    project_1.add_experiment(experiment="Exp01", units="real")
    project_1.add_experiment(experiment="Exp02", units=custom_units)

    project_2 = mds.Project()

    for key, val in project_2.experiments["Exp01"].simulation_data.items():
        assert val == units_real()[key]

    for key, val in project_2.experiments["Exp02"].simulation_data.items():
        assert val == custom_units[key]
