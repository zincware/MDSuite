"""
Class for the calculation of the Green-Kubo ionic conductivity.

Summary
This module contains the code for the Green-Kubo ionic conductivity class. This class is called by the
Experiment class and instantiated when the user calls the Experiment.green_kubo_ionic_conductivity method.
The methods in class can then be called by the Experiment.green_kubo_ionic_conductivity method and all necessary
calculations performed.
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
plt.style.use('bmh')
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboIonicConductivity(Calculator):
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

    def __init__(self, obj, plot=False, data_range=500, x_label='Time (s)', y_label=r'JACF ($C^{2}\cdot m^{2}/s^{2}$)',
                 save=True, analysis_name='green_kubo_ionic_conductivity'):
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
        self.database_group = 'ionic_conductivity'  # Which database group to save the data in

        # Time array
        self.time = np.linspace(0.0, data_range * self.parent.time_step * self.parent.sample_rate, data_range)
        self.correlation_time = 100  # correlation time of the system current.

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
                ionic current of the system as a vector of shape (n_confs, 3)
        """

        species_charges = [self.parent.species[atom]['charge'][0] for atom in self.parent.species]  # build charge array

        system_current = np.zeros((self.batch_size['Parallel']*self.data_range, 3))  # instantiate the current array
        # Calculate the total system current
        for i in range(len(velocity_matrix)):
            system_current += np.array(np.sum(velocity_matrix[i][:, 0:], axis=0)) * species_charges[i]

        return system_current

    def _calculate_ionic_conductivity(self):
        """
        Calculate the ionic conductivity in the system
        """

        # Calculate the prefactor
        numerator = (elementary_charge ** 2) * (self.parent.units['length'] ** 2)
        denominator = 3 * boltzmann_constant * self.parent.temperature * self.parent.volume * \
                      (self.parent.units['length'] ** 3) * self.data_range * self.parent.units['time']
        prefactor = numerator / denominator

        sigma = []                                          # define an empty sigma list
        parsed_autocorrelation = np.zeros(self.data_range)  # Define the parsed array

        for i in tqdm(range(int(self.n_batches['Parallel'])), ncols=70):                 # loop over batches
            batch = self._calculate_system_current(velocity_matrix=self._load_batch(i))  # get the ionic current batch
            for start_index in range(self.batch_loop):                                   # loop over ensembles in batch
                start = int(start_index*self.data_range + self.correlation_time)         # get start index
                stop = int(start + self.data_range)                                      # get stop index
                system_current = np.array(batch)[start:stop]                             # load data from batch array

                jacf = np.zeros(2 * self.data_range - 1)                                 # Define the empty jacf array

                # Calculate the current autocorrelation
                jacf += (signal.correlate(system_current[:, 0],
                                          system_current[:, 0],
                                          mode='full', method='auto') +
                         signal.correlate(system_current[:, 1],
                                          system_current[:, 1],
                                          mode='full', method='auto') +
                         signal.correlate(system_current[:, 2],
                                          system_current[:, 2],
                                          mode='full', method='auto'))

                jacf = jacf[int((len(jacf) / 2)):]  # Cut the negative part of the current autocorrelation
                parsed_autocorrelation += jacf      # update parsed function
                sigma.append(prefactor * np.trapz(jacf, x=self.time))  # Update the conductivity array

        # update the experiment class
        self.parent.ionic_conductivity["Green-Kubo"] = [np.mean(sigma) / 100, (np.std(sigma)/np.sqrt(len(sigma)))/100]

        plt.plot(self.time * self.parent.units['time'], parsed_autocorrelation)  # Add a plot

        parsed_autocorrelation /= max(parsed_autocorrelation)  # Get the normalized autocorrelation plot data

        # save the data if necessary
        if self.save:
            self._save_data(f'{self.analysis_name}', [self.time, parsed_autocorrelation])

        if self.plot:
            self._plot_data()  # Plot the data if necessary

    def run_analysis(self):
        """
        call relevant methods and run analysis
        """

        self._autocorrelation_time()          # get the autocorrelation time
        self._collect_machine_properties()    # collect machine properties and determine batch size
        self._calculate_batch_loop()          # Update the batch loop attribute
        self._calculate_ionic_conductivity()  # calculate the ionic conductivity
