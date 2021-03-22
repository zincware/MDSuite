import numpy as np
import h5py as hf
import matplotlib.pyplot as plt
colors = ['C0', 'C1']
symbols = ['d', 'o']


# Plot the integrated heat current
datasets = ['Integrated_heat_current_kinaci', 'Integrated_heat_current']
labels = ['kinaci', 'normal_eh']

fig1, ax1 = plt.subplots()
for dataset, label, color in zip(datasets, labels, colors):
    path = f'{dataset}/{dataset}'
    with hf.File('databases/database.hdf5', "r+") as database:
        data = database[path][np.s_[:]]

    slice = np.s_[:,:]

    slice_property = data[slice]
    ax1.plot(slice_property.sum(axis=1), alpha=0.5, markevery=50, label=label, color=color)



# Plot the average system displacement. (straight line)
datasets = ['einstein_helfand_thermal_conductivity_kinaci', 'einstein_helfand_thermal_conductivity']
labels = ['kinaci', 'normal_eh']
slopes = [10484.337, 1565.3]

fig2, ax2 = plt.subplots()
for dataset, label, color, slope in zip(datasets, labels, colors, slopes):
    path = f'thermal_conductivity/{dataset}'
    with hf.File('databases/analysis_data.hdf5', "r+") as database:
        data = database[path][np.s_[:]]
    x = data[0,:]
    y = data[1,:]

    ax2.plot(x, y, alpha=0.5, markevery=20, label=label, color=color, marker=symbols[0])
    ax2.plot(x, x*slope, alpha=0.5, markevery=20, color=color)







plt.show()