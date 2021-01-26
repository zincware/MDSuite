"""
Class for the calculation of the Green-Kubo thermal conductivity.

Summary
-------
This module contains the code for the Green-Kubo thermal conductivity class. This class is called by the
Experiment class and instantiated when the user calls the Experiment.green_kubo_thermal_conductivity method.
The methods in class can then be called by the Experiment.green_kubo_thermal_conductivity method and all necessary
calculations performed.
"""

import warnings

# Python standard packages
import matplotlib.pyplot as plt
import numpy as np
# Import user packages
from tqdm import tqdm

from mdsuite.calculators.calculator import Calculator

# Import MDSuite modules

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
plt.style.use('bmh')
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboThermalConductivity(Calculator):
    """
    Class for the Green-Kubo Thermal conductivity implementation

    Attributes
    ----------
    obj :  object
            Experiment class to call from
    plot : bool
            if true, plot the data
    data_range :
            Number of configurations to use in each ensemble
    save :
            If true, data will be saved after the analysis
    x_label : str
            X label of the data when plotted
    y_label : str
            Y label of the data when plotted
    analysis_name : str
            Name of the analysis
    time : np.array
            Array of the time.
    correlation_time : int
            Correlation time of the property being studied. This is used to ensure ensemble sampling is only performed
            on uncorrelated samples. If this is true, the error extracted form the calculation will be correct.
    """

    def __init__(self, obj, plot=False, data_range=500, x_label='Time (s)', y_label='JACF ($C^{2}\cdotm^{2}/s^{2}$)',
                 save=True, analysis_name='green_kubo_thermal_conductivity'):
        """
        Class for the Green-Kubo Thermal conductivity implementation

        Attributes
        ----------
        obj :  object
                Experiment class to call from
        plot : bool
                if true, plot the data
        data_range :
                Number of configurations to use in each ensemble
        save :
                If true, data will be saved after the analysis
        x_label : str
                X label of the data when plotted
        y_label : str
                Y label of the data when plotted
        analysis_name : str
                Name of the analysis
        """
        super().__init__(obj, plot, save, data_range, x_label, y_label, analysis_name)

        self.number_of_configurations = self.parent.number_of_configurations - self.parent.number_of_configurations % \
                                        self.parent.batch_size
        self.time = np.linspace(0.0, data_range * self.parent.time_step * self.parent.sample_rate, data_range)
        self.loop_range = self.number_of_configurations - data_range - 1
        self.correlation_time = 1
        self.database_group = 'thermal_conductivity'  # Which database group to save the data in
        self.loaded_properties = {'Velocities', 'Stress', 'ke', 'pe'}  # property to be loaded for the analysis
        self.loaded_property = 'Velocities'
        self.parallel = True

    def _autocorrelation_time(self):
        """
        calculate the current autocorrelation time for correct sampling
        """
        pass

    def _calculate_system_current(self, i):
        """
        Calculate the thermal current of the system

        Returns
        -------
        system_current : np.array
                thermal current of the system as a vector of shape (n_confs, 3)
        """

        velocity_matrix = self._load_batch(i, "Velocities")  # Load the velocity matrix
        stress_tensor = self._load_batch(i, "Stress", sym_matrix=True)
        pe = self._load_batch(i, "PE", scalar=True)
        ke = self._load_batch(i, "KE", scalar=True)

        # define phi as product stress tensor * velocity matrix.
        # It is done by components to take advantage of the symmetric matrix.
        phi_x = np.multiply(stress_tensor[:, :, 0], velocity_matrix[:, :, 0]) + \
                np.multiply(stress_tensor[:, :, 3], velocity_matrix[:, :, 1]) + \
                np.multiply(stress_tensor[:, :, 4], velocity_matrix[:, :, 2])
        phi_y = np.multiply(stress_tensor[:, :, 3], velocity_matrix[:, :, 0]) + \
                np.multiply(stress_tensor[:, :, 1], velocity_matrix[:, :, 1]) + \
                np.multiply(stress_tensor[:, :, 5], velocity_matrix[:, :, 2])
        phi_z = np.multiply(stress_tensor[:, :, 4], velocity_matrix[:, :, 0]) + \
                np.multiply(stress_tensor[:, :, 5], velocity_matrix[:, :, 1]) + \
                np.multiply(stress_tensor[:, :, 2], velocity_matrix[:, :, 2])

        phi = np.dstack([phi_x, phi_y, phi_z])

        phi_sum_atoms = phi.sum(axis=0) / self.parent.units['NkTV2p']  # factor for units lammps nktv2p

        # ke_total = np.sum(ke, axis=0) # to check it was the same, can be removed.
        # pe_total = np.sum(pe, axis=0)

        energy = ke + pe

        energy_velocity = energy[:, :, None] * velocity_matrix
        energy_velocity_atoms = energy_velocity.sum(axis=0)

        system_current = energy_velocity_atoms - phi_sum_atoms  # returns the same values as in the compute flux of lammps

        return system_current

    def _calculate_thermal_conductivity(self):
        """
        Calculate the thermal conductivity in the system
        """

        # prepare the prefactor for the integral
        numerator = 1
        denominator = 3 * (self.data_range - 1) * self.parent.temperature ** 2 * self.parent.units['boltzman'] \
                      * self.parent.volume  # we use boltzman constant in the units provided.

        # not sure why I need the /2 in data range...
        prefactor = numerator / denominator

        sigma, parsed_autocorrelation = self.convolution_operation(type_batches='Parallel')

        # convert to SI units.
        prefactor_units = self.parent.units['energy'] / self.parent.units['length'] / self.parent.units['time']
        sigma = prefactor * prefactor_units * np.array(sigma)

        self.parent.thermal_conductivity["Green-Kubo"] = np.mean(sigma)

        plt.plot(self.time, parsed_autocorrelation)  # Add a plot

        parsed_autocorrelation /= max(parsed_autocorrelation)  # Get the normalized autocorrelation plot data
        if self.save:
            self._save_data(f'{self.analysis_name}', [self.time, parsed_autocorrelation])

        if self.plot:
            self._plot_data()  # Plot the data if necessary

    def run_analysis(self):
        """ Run thermal conductivity calculation analysis

        The thermal conductivity is computed at this step.
        """
        self._autocorrelation_time()  # get the autocorrelation time
        self._collect_machine_properties()    # collect machine properties and determine batch size
        self._calculate_batch_loop()          # Update the batch loop attribute
        status = self._check_input()          # Check for bad input
        if status == -1:
            return
        else:
            self._calculate_thermal_conductivity()  # calculate the singular diffusion coefficients
