import numpy as np
import pytest

import mdsuite
from mdsuite.database.simulation_database import (
    PropertyInfo,
    SpeciesInfo,
    TrajectoryChunkData,
    TrajectoryMetadata,
)
from mdsuite.file_io import script_input

mdsuite.config.memory_fraction = 1.0
mdsuite.config.memory_scaling_test = True


@pytest.fixture(params=[50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 10000])
def project(tmp_path, request) -> mdsuite.Project:
    """Build a MDSuite Project with dummy data

    This creates a project with data for velocities and positions
    generated randomly for 100 configurations and a variable size of
    particles given by the fixture definition
    """
    n_configs = 100
    n_parts = request.param
    n_dims = 3
    time_step = 0.1
    sp_name = "species"
    positions = np.random.rand(*(n_configs, n_parts, n_dims))
    velocities = np.random.rand(*(n_configs, n_parts, n_dims))

    properties = [
        PropertyInfo(name="Positions", n_dims=n_dims),
        PropertyInfo(name="Velocities", n_dims=n_dims),
    ]

    species_list = [
        SpeciesInfo(name=sp_name, n_particles=n_parts, mass=1234, properties=properties)
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


@pytest.mark.memory
def test_adf(project):
    _ = project.run.AngularDistributionFunction(number_of_configurations=-1, plot=False)
