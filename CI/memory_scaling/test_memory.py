import numpy as np
import pytest

import mdsuite
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.database.simulation_database import (
    SpeciesInfo,
    TrajectoryChunkData,
    TrajectoryMetadata,
)
from mdsuite.file_io import script_input

mdsuite.config.memory_fraction = 1.0
mdsuite.config.memory_scaling_test = True


def get_project(tmp_path, n_configs, n_parts) -> mdsuite.Project:
    """Build a MDSuite Project with dummy data

    This creates a project with data for velocities and positions
    generated randomly for 100 configurations and a variable size of
    particles given by the fixture definition
    """
    n_dims = 3
    time_step = 0.1
    sp_name = "species"
    positions = np.random.rand(*(n_configs, n_parts, n_dims))
    velocities = np.random.rand(*(n_configs, n_parts, n_dims))

    species_list = [
        SpeciesInfo(
            name=sp_name,
            n_particles=n_parts,
            mass=1234,
            properties=[mdsuite_properties.positions, mdsuite_properties.velocities],
        )
    ]

    metadata = TrajectoryMetadata(
        species_list=species_list,
        n_configurations=n_configs,
        sample_rate=1,
        box_l=3 * [1.1],
    )
    data = TrajectoryChunkData(species_list=species_list, chunk_size=n_configs)
    data.add_data(positions, 0, sp_name, "Positions")
    data.add_data(velocities, 0, sp_name, "Velocities")

    proc = script_input.ScriptInput(data=data, metadata=metadata, name="test_name")

    project = mdsuite.Project(name="test_proj", storage_path=tmp_path)
    project.add_experiment(name="test_experiment", timestep=time_step)
    exp = project.experiments["test_experiment"]
    exp.add_data(proc)

    return project


@pytest.mark.parametrize("n_parts", [x for x in range(10, 300, 10)])
@pytest.mark.memory
def test_adf(tmp_path, n_parts):
    project = get_project(tmp_path, n_configs=5, n_parts=n_parts)
    _ = project.run.AngularDistributionFunction(number_of_configurations=2, plot=False)


@pytest.mark.parametrize("n_parts", [x for x in range(100, 12000, 200)])
@pytest.mark.memory
def test_rdf(tmp_path, n_parts):
    project = get_project(tmp_path, n_configs=15, n_parts=n_parts)
    _ = project.run.RadialDistributionFunction(number_of_configurations=10, plot=False)


@pytest.mark.parametrize("n_configs", [x for x in range(100, 12000, 200)])
@pytest.mark.memory
def test_einstein_diffusion(tmp_path, n_configs):
    project = get_project(tmp_path, n_configs=n_configs, n_parts=100)
    _ = project.run.EinsteinDiffusionCoefficients(plot=False)


@pytest.mark.parametrize("n_configs", [x for x in range(500, 12000, 200)])
@pytest.mark.memory
def test_gk_diffusion(tmp_path, n_configs):
    project = get_project(tmp_path, n_configs=n_configs, n_parts=100)
    _ = project.run.GreenKuboDiffusionCoefficients(plot=False)
