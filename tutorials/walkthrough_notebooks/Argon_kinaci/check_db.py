import os
import numpy as np
import h5py as hf
import matplotlib.pyplot as plt

path = '1/Unwrapped_Positions'
with hf.File('Argon/databases/database.hdf5', "r+") as database:
    data = database[path][np.s_[:]]

slice = np.s_[[0,5,20,200], :,1]

fig, ax = plt.subplots()
slice_property = data[slice]
print(slice_property.shape)
ax.plot(slice_property.T, 'o')



path = 'Integrated_heat_current_kinaci/Integrated_heat_current_kinaci'
with hf.File('database.hdf5', "r+") as database:
    data = database[path][np.s_[:]]

slice = np.s_[:,:]

fig2, ax2 = plt.subplots()
slice_property = data[slice]
print(slice_property.shape)
ax2.plot(slice_property.sum(axis=1), '-d', label='Kinaci', alpha=0.5, markevery=20)

# path = 'Integrated_heat_current/Integrated_heat_current'
# with hf.File('database.hdf5', "r+") as database:
#     data = database[path][np.s_[:]]
#
# slice = np.s_[:,:]
# ax2.plot(slice_property.sum(axis=1), '-o', label='Normal', alpha=0.5, markevery=20)

ax2.legend()


fig.savefig('positions_unwrapped.png')
fig2.savefig('Integrated_heat_current_kinaci.png')