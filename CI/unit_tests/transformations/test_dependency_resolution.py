import os

import numpy as np

import mdsuite as mds
import mdsuite.database.simulation_database
import mdsuite.file_io.script_input
from mdsuite.database.simulation_data_class import mdsuite_properties


def test_transf(tmp_path):
    """
    check if positions are automatically unwrapped if needed in another transformation
    """
    os.chdir(tmp_path)

    n_config = 100
    n_part = 10
    t_step = 0.123
    box_l = [1.1, 2.2, 3.3]

    project = mds.Project()
    project.add_experiment(name="TestExp", timestep=t_step)
    exp = project.experiments["TestExp"]

    pos_prop = mdsuite_properties.positions

    box_im_prop = mdsuite_properties.box_images
    species = mdsuite.database.simulation_database.SpeciesInfo(
        "test_species", n_part, [pos_prop, box_im_prop]
    )
    mdata = mdsuite.database.simulation_database.TrajectoryMetadata(n_config, [species])

    pos = np.random.random((n_config, n_part, pos_prop.n_dims))
    box_im = np.random.randint(-100, 100, size=(n_config, n_part, box_im_prop.n_dims))

    chunk = mdsuite.database.simulation_database.TrajectoryChunkData([species], n_config)
    chunk.add_data(pos, 0, species.name, pos_prop.name)
    chunk.add_data(box_im, 0, species.name, box_im_prop.name)

    np_read = mdsuite.file_io.script_input.ScriptInput(chunk, mdata, "test_reader")
    exp.add_data(np_read)
    exp.box_array = box_l

    exp.run.VelocityFromPositions()
    unwrapped_pos = exp.load_matrix(
        property_name=mdsuite_properties.unwrapped_positions.name, species=[species.name]
    )[f"{species.name}/{mdsuite_properties.unwrapped_positions.name}"]

    assert len(unwrapped_pos) == n_part
