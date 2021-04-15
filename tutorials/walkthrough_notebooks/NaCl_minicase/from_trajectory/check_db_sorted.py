import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
import numpy as np
import h5py as hf
import matplotlib.pyplot as plt
from plot_python_vki import  apply_style
import mdsuite as mds  # Cool name for professionalism purposes

case = 'NaCl_sorted'

new_case = True

if new_case:
    try:
        shutil.rmtree(case)
    except FileNotFoundError:
        pass

argon = mds.Experiment(analysis_name=case, time_step=1, temperature=70.0, units='real')

if new_case:
    argon.add_data(trajectory_file='../gk_data_sorted.lmp_traj')

argon.run_computation.EinsteinHelfandThermalConductivity(data_range=6, plot=True, correlation_time=1)

plt.figure()
apply_style()
plt.style.use({"lines.markeredgewidth": 0, 'figure.figsize': [16, 9]})

folder = case

path = '1/Positions'
with hf.File(f'{folder}/databases/database.hdf5', "r+") as database:
    data = database[path][np.s_[:]]

slice = np.s_[:, :, 1]

sliced_x = data[slice]

plt.plot(sliced_x[0:10].T, 'o')

plt.savefig(f"{folder}/positions.png")
plt.show()
