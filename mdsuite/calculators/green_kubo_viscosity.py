"""
Class for the calculation of the Green-Kubo ionic conductivity.

Summary
This module contains the code for the Green-Kubo viscsity class. This class is called by the
"""

import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import warnings

# Import user packages
from tqdm import tqdm

# Import MDSuite modules
from mdsuite.utils.units import boltzmann_constant, elementary_charge

from mdsuite.calculators.calculator import Calculator

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboViscosity(Calculator):
    """ Class for the Green-Kubo ionic conductivity implementation

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
    loaded_property : str
            Property loaded from the database for the analysis
    batch_loop : int
            Number of ensembles in each batch
    time : np.array
            Array of the time.
    correlation_time : int
            Correlation time of the property being studied. This is used to ensure ensemble sampling is only performed
            on uncorrelated samples. If this is true, the error extracted form the calculation will be correct.
    """

    def __init__(self, obj, plot=False, data_range=500, x_label='Time (s)', y_label=r'SACF ($C^{2}\cdot m^{2}/s^{2}$)',
                 save=True, analysis_name='green_kubo_viscosity'):
        """

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

        self.loaded_property = 'Velocities'         # property to be loaded for the analysis
        self.batch_loop = None                      # Number of ensembles in each batch
        self.parallel = True                        # Set the parallel attribute
        self.tensor_choice = False                  # Load data as a tensor
        self.database_group = 'viscosity'           # Which database group to save the data in

        # Time array
        self.time = np.linspace(0.0, data_range * self.parent.time_step * self.parent.sample_rate, data_range)
        self.correlation_time = 1  # correlation time of the system current.
        
    def _autocorrelation_time(self):
        """
        calculate the current autocorrelation time for correct sampling
        """
        pass

    def _calculate_system_current(self, velocity_matrix):
        """
        Calculate the ionic current of the system

        Parameters
        ----------
        velocity_matrix : np.array
                tensor of system velocities for use in the current calculation

        Returns
        -------
        system_current : np.array
                thermal current of the system as a vector of shape (number_of_configurations, 3)
        """

        # velocity_matrix = self._load_batch(i, "Velocities")  # Load the velocity matrix
        stress_tensor = self.load_batch(i, "Stress", sym_matrix=True)

        # we take the xy, xz, and yz components (negative)
        phi_x = -stress_tensor[:, :, 3]
        phi_y = -stress_tensor[:, :, 4]
        phi_z = -stress_tensor[:, :, 5]

        phi = np.dstack([phi_x, phi_y, phi_z])

        phi_sum_atoms = phi.sum(axis=0)

        system_current = phi_sum_atoms  # returns the same values as in the compute flux of lammps

        return system_current

    def _calculate_viscosity(self):
        """
        Calculate the viscosity of the system
        """

        # prepare the prefactor for the integral
        # Since lammps gives the stress in pressure*volume, then we need to add to the denominator volume**2,
        # this is why the numerator becomes 1, and volume appears in the denominator.
        numerator = 1  # self.parent.volume
        denominator =  3 * (self.data_range - 1) * self.parent.temperature * self.parent.units[
            'boltzman'] * self.parent.volume  # we use boltzman constant in the units provided.

        prefactor = numerator / denominator

        sigma, parsed_autocorrelation = self.convolution_operation(type_batches='Parallel')

        # convert to SI units.
        prefactor_units = self.parent.units['pressure'] ** 2 * self.parent.units['length'] ** 3 * self.parent.units[
            'time'] / self.parent.units['energy']
        sigma = prefactor * prefactor_units * np.array(sigma)

        self.parent.viscosity["Green-Kubo"] = np.mean(sigma)
        self._update_properties_file(data=str(np.mean(sigma)))

        plt.plot(self.time, parsed_autocorrelation)  # Add a plot

        parsed_autocorrelation /= max(parsed_autocorrelation)  # Get the normalized autocorrelation plot data

        if self.save:
            self._save_data(f'{self.analysis_name}', [self.time, parsed_autocorrelation])

        if self.plot:
            self._plot_data()  # Plot the data if necessary

    def run_analysis(self):
        """
        call relevant methods and run analysis
        """

        self._autocorrelation_time()          # get the autocorrelation time
        self.collect_machine_properties()    # collect machine properties and determine batch size
        self._calculate_batch_loop()          # Update the batch loop attribute
        status = self._check_input()          # Check for bad input
        if status == 0:
            return
        else:
            self._calculate_ionic_conductivity()  # calculate the ionic conductivity
