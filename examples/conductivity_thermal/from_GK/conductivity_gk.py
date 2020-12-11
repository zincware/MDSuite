import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import mdsuite as mds # Cool name for professionalism purposes

argon = mds.Experiment(analysis_name="Argon", timestep=4, temperature=70.0, units='real', )

argon.add_data(trajectory_file='../gk_data.txt')
#
# argon.einstein_diffusion_coefficients(plot=True, data_range=50)
# argon.green_kubo_diffusion_coefficients(plot=True, data_range=50)
#
# print(argon.diffusion_coefficients)