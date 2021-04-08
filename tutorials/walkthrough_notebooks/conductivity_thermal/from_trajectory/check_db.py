import os
import numpy as np
import h5py as hf
import matplotlib.pyplot as plt

folder = 'Argon'

path = '1/Unwrapped_Positions'
with hf.File(f'{folder}/databases/database.hdf5', "r+") as database:
    data = database[path][np.s_[:]]

slice = np.s_[:, :,1]

sliced_x = data[slice]

plt.plot(sliced_x[0:10].T, 'o')

plt.savefig(f"{folder}/positions.png")
plt.show()