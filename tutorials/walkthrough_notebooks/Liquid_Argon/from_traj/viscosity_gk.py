import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil

import mdsuite as mds # Cool name for professionalism purposes

try:
    shutil.rmtree('Argon')
except FileNotFoundError:
    pass


argon = mds.Experiment(analysis_name="Argon", time_step=2, temperature=94.4, units='real')


argon.add_data(trajectory_file='../gk_data.lmp_traj')

#argon.run_computation('GreenKuboThermalConductivity', data_range=19980, plot=True)
argon.run_computation('GreenKuboViscosity', data_range=19980, plot=True)
print(argon.results)
