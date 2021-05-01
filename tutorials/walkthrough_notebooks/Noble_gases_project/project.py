import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mdsuite as mds  # Import the mdsuite python package
from mdsuite.plot_style.plot_style import apply_style

apply_style()

new_project = False
if new_project:
    try:
        shutil.rmtree('Gases_MDSuite_Project')
    except FileNotFoundError:
        pass

gases = mds.Project(name="Gases", storage_path="./")

gases.add_description("Project to analyze and compare gases")

gases.add_experiment(experiment="Argon_135",
                     timestep=3,
                     temperature=135.6,
                     units='real')

gases.add_experiment(experiment="Argon_93",
                     timestep=3,
                     temperature=92.97,
                     units='real')

gases.add_experiment(experiment="Xenon_229",
                     timestep=3,
                     temperature=229.0,
                     units='real')

gases.add_experiment(experiment="Xenon_176",
                     timestep=3,
                     temperature=176.2,
                     units='real')

print(gases)

if new_project:
    # you can access directly the experiment as an attribute
    gases.Argon_93.add_data(trajectory_file='Argon/0.942_gcm3_135.6_K/gk_data.lmp_traj')
    gases.Xenon_229.add_data(trajectory_file='Xenon/2.243_gcm3_229.0_K/gk_data.lmp_traj')
    gases.Xenon_176.add_data(trajectory_file='Xenon/2.916_gcm3_176.2_K/gk_data.lmp_traj')
    gases.Argon_135.add_data(trajectory_file='Argon/1.375_gcm3_92.97_K/gk_data.lmp_traj')

# Then for each of them, we can run the conductivity computation
if new_project:
    for experiment_name, experiment_class in gases.experiments.items():
        experiment_class.run_computation.GreenKuboThermalConductivity(data_range=500, plot=True, correlation_time=1)

results = gases.Argon_93.results
print(results)

# get it for all the cases
results_k = gases.get_results('green_kubo_thermal_conductivity')
temperatures = gases.get_attribute('temperature')

# put the results in a dataframe
dict_results = pd.DataFrame.from_dict(results_k)
dict_results = dict_results.append(temperatures, ignore_index=True)
dict_results = dict_results.rename(index={0: 'k', 1: 'k_err', 2: 'temperature'})
print(dict_results)

# plot the results
fig, ax = plt.subplots()

dfs = {'argon': dict_results.iloc[:, 0:2], 'xenon': dict_results.iloc[:, 2:]}

colors = ['C0', 'C1']

for idx, (key, val) in enumerate(dfs.items()):
    print(key)
    temperature = val.loc['temperature', :].values
    k = val.loc['k', :].values
    k_err = val.loc['k_err', :].values
    ax.errorbar(temperature, k, yerr=k_err, linestyle="--", marker='o', color=colors[idx])

ref_data = {'argon': {93: 0.121, 135: 0.058}, 'xenon': {176: 0.075, 229: 0.0397}}

for idx, (key, val) in enumerate(ref_data.items()):
    temperature = []
    k = []
    for temp, conduct in val.items():
        temperature.append(temp)
        k.append(conduct)

    ax.plot(temperature, k, linestyle="-", marker='s', color=colors[idx])

ax.plot(np.nan, np.nan, linestyle="--", color='k', marker='o', label='Ref.')
ax.plot(np.nan, np.nan, linestyle="-", color='k', marker='s', label='MDSuite')

ax.plot(np.nan, np.nan, linestyle="-", color='C0', label='Argon')
ax.plot(np.nan, np.nan, linestyle="-", color='C1', label='Xenon')

ax.legend()

ax.set_xlabel("Temperature, K")
ax.set_ylabel("Thermal Conductivity, W/m/K")

plt.show()
