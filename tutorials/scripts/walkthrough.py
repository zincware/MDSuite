"""
Quick walkthrough script for MDSuite
"""
import mdsuite as mds

if __name__ == "__main__":
    # Add project
    Molten_NaCl = mds.Project(name="Molten_NaCl", storage_path="./")
    # Project description (optional)
    Molten_NaCl.add_description("Best project ever!")
    # Add experiment to project
    Molten_NaCl.add_experiment(experiment="NaCl_1400K",
                               timestep=0.002,
                               temperature=1400.0,
                               units='metal')

    # Create experiment object
    NaCl_1400K = Molten_NaCl.experiments['NaCl_1400K']  # The nice way

    # Add tensor_values to the experiment
    NaCl_1400K.add_data(trajectory_file='../data/trajectory_files/NaCl_1400K.dump')
