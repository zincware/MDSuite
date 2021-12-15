import os

import numpy as np

import mdsuite as mds
import mdsuite.database.simulation_data_class
import mdsuite.database.simulation_database
import mdsuite.file_io.script_input
import mdsuite.transformations.velocity_from_positions


def test_transf(tmp_path):
    os.chdir(tmp_path)

    n_config = 100
    n_part = 10
    t_step = 0.123

    project = mds.Project()
    project.add_experiment(name="TestExp", timestep=t_step)
    exp = project.experiments["TestExp"]

    pos_prop = (
        mdsuite.database.simulation_data_class.mdsuite_properties.unwrapped_positions
    )
    species = mdsuite.database.simulation_database.SpeciesInfo(
        "test_species", n_part, [pos_prop]
    )
    mdata = mdsuite.database.simulation_database.TrajectoryMetadata(n_config, [species])

    pos = np.random.random((n_config, n_part, pos_prop.n_dims))
    chunk = mdsuite.database.simulation_database.TrajectoryChunkData([species], n_config)
    chunk.add_data(pos, 0, species.name, pos_prop.name)

    np_read = mdsuite.file_io.script_input.ScriptInput(chunk, mdata, "test_reader")
    exp.add_data(np_read)

    trfo = mdsuite.transformations.velocity_from_positions.VelocityFromPositions(exp)
    trfo.run_transformation()
    vels_mds = exp.load_matrix(
        property_name="Velocities_From_Positions", species=[species.name]
    )[f"{species.name}/Velocities_From_Positions"]

    pos = np.swapaxes(pos, 0, 1)
    vels_numpy = (pos[:, 1:, :] - pos[:, :-1, :]) / t_step
    last_vels = vels_numpy[:, -1, :]
    vels_numpy = np.concatenate((vels_numpy, last_vels[:, None, :]), axis=1)

    np.testing.assert_almost_equal(vels_mds, vels_numpy, decimal=4)
