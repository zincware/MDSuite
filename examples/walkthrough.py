import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import mdsuite as mds # Cool name for professionalism purposes

NaCl_1400K = mds.Experiment(analysis_name="NaCl", timestep=0.002, temperature=80.0, units='metal')

#NaCl_1400K.add_data(trajectory_file='trajectory_files/NaCl_1400K.dump')

#aCl_1400K.unwrap_coordinates()

#NaCl_1400K.print_class_attributes()

NaCl_1400K.einstein_diffusion_coefficients(plot=True, data_range=150)
#NaCl_1400K.green_kubo_diffusion_coefficients(plot=True, data_range=50)

print(NaCl_1400K.diffusion_coefficients)

NaCl_1400K.write_xyz(property="Unwrapped_Positions")
