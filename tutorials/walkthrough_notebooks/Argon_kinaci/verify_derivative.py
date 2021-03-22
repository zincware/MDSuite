import numpy as np
import h5py as hf
import matplotlib.pyplot as plt
colors = ['C0', 'C1']
symbols = ['d', 'o']
fig2, ax2 = plt.subplots()

factor = 10000 # factor to scale the derivatives.

# Plot the flux J
path = f'Thermal_Flux/Thermal_Flux'
with hf.File('Argon/databases/database.hdf5', "r+") as database:
    data = database[path][np.s_[:]]

slice = np.s_[:, :]

slice_property = data[slice]
ax2.plot(slice_property.sum(axis=1), alpha=0.5, markevery=50, label="J", color='k')

# Plot the derivatives of R

datasets = ['Integrated_heat_current_kinaci', 'Integrated_heat_current']
labels = ['kinaci', 'normal_eh']

for dataset, label, color in zip(datasets, labels, colors):
    path = f'{dataset}/{dataset}'
    with hf.File('Argon/databases/database.hdf5', "r+") as database:
        data = database[path][np.s_[:]]

    slice = np.s_[:,:]

    slice_property = data[slice]
    data = slice_property.sum(axis=1)
    derivative = np.diff(data)/factor
    ax2.plot(derivative, alpha=0.5, markevery=50, label=label, color=color)

plt.show()