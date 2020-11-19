import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import mdsuite as mds # Cool name for professionalism purposes

NaCl_1400K = mds.Experiment(analysis_name="NaCl_1400K", timestep=0.002, temperature=1400.0, units='metal')

#NaCl_1400K.add_data(trajectory_file='trajectory_files/NaCl_1400K.dump')

NaCl_1400K.einstein_diffusion_coefficients(plot=True, data_range=100)

