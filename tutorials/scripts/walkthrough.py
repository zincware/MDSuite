import mdsuite as mds  # Import the mdsuite python package


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

# Add data to the experiment
NaCl_1400K.add_data(trajectory_file='../data/trajectory_files/NaCl_1400K.dump')


NaCl_1400K.unwrap_coordinates()
NaCl_1400K.collect_memory_information()
NaCl_1400K.save_class()

print('\n number of configs \n', NaCl_1400K.number_of_configurations)

#################
# Analysis part #
#################

# Do the diffusion analysis with your experiment
NaCl_1400K.einstein_diffusion_coefficients(plot=True, data_range=50)

# Do the radial distribution function analysis with your experiment
#NaCl_1400K.radial_distribution_function(plot=True, stop=100)


# print the diffusion coefficients
print(NaCl_1400K.diffusion_coefficients['Einstein']['Singular'])
