"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Tests for the experiment database
"""
import os
from tempfile import TemporaryDirectory
import pytest
import mdsuite as mds

temp_dir = TemporaryDirectory()
cwd = os.getcwd()


@pytest.fixture(autouse=True)
def prepare_env():
    temp_dir = TemporaryDirectory()
    os.chdir(temp_dir.name)

    yield

    os.chdir(cwd)
    temp_dir.cleanup()


def test_project_temperature():
    """Test that the project description is stored correctly in the database"""

    project_1 = mds.Project()
    project_1.add_experiment(experiment="Exp01")
    project_1.experiments['Exp01'].temperature = 9000

    project_2 = mds.Project()
    project_2.load_experiments('Exp01')
    assert project_2.experiments['Exp01'].temperature == 9000


def test_project_time_step():
    """Test that the project description is stored correctly in the database"""

    project_1 = mds.Project()
    project_1.add_experiment(experiment="Exp01")
    project_1.experiments['Exp01'].time_step = 1

    project_2 = mds.Project()
    project_2.load_experiments('Exp01')
    assert project_2.experiments['Exp01'].time_step == 1


def test_project_unit_system():
    """Test that the project description is stored correctly in the database"""

    project_1 = mds.Project()
    project_1.add_experiment(experiment="Exp01")
    project_1.experiments['Exp01'].unit_system = "real"

    project_2 = mds.Project()
    project_2.load_experiments('Exp01')
    assert project_2.experiments['Exp01'].unit_system == "real"


def test_project_number_of_configurations():
    """Test that the project description is stored correctly in the database"""

    project_1 = mds.Project()
    project_1.add_experiment(experiment="Exp01")
    project_1.experiments['Exp01'].number_of_configurations = 100

    project_2 = mds.Project()
    project_2.load_experiments('Exp01')
    assert project_2.experiments['Exp01'].number_of_configurations == 100

def test_project_number_of_atoms():
    """Test that the project description is stored correctly in the database"""

    project_1 = mds.Project()
    project_1.add_experiment(experiment="Exp01")
    project_1.experiments['Exp01'].number_of_atoms = 100

    project_2 = mds.Project()
    project_2.load_experiments('Exp01')
    assert project_2.experiments['Exp01'].number_of_atoms == 100
