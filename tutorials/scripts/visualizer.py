"""
Tutorial Name: Visualizer

Description: This script goes over how to use the MDSuite visualizer to look at your trajectories either for fun or
for real interest in a specific phenomenon.
"""

import mdsuite as mds

def main():
    """
    Boilerplate main function

    Returns
    -------
    Nothing. Runs the example.
    """
    # Add project
    Molten_NaCl = mds.Project(name="Molten_NaCl", storage_path="./")
    # Project description (optional)
    Molten_NaCl.add_description("Best project ever!")
    # Add experiment to project
    Molten_NaCl.add_experiment(experiment_name="NaCl_1400K",
                               timestep=0.002,
                               temperature=1400.0,
                               units='metal')
    # Create experiment object
    NaCl_1400K = Molten_NaCl.experiments['NaCl_1400K']  # The nice way
    # Add tensor_values to the experiment
    NaCl_1400K.add_data(trajectory_file='../data/trajectory_files/NaCl_1400K.dump')
    NaCl_1400K.perform_transformation('UnwrapCoordinates')
    NaCl_1400K.run_visualization()  # run the visualizer


if __name__ == "__main__":
    """
    Standard boilerplate
    """
    main()


