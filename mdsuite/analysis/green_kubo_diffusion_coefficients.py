"""
Class for the calculation of the Green-Kubo diffusion coefficients.

Author: Samuel Tovey

Description: This module contains the code for the Einstein diffusion coefficient class. This class is called by the
Experiment class and instantiated when the user calls the Experiment.einstein_diffusion_coefficients method.
The methods in class can then be called by the Experiment.green_kubo_diffusion_coefficients method and all necessary
calculations performed.
"""

# Python standard packages
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import warnings

# Import user packages
from tqdm import tqdm
import itertools
import multiprocessing as mp

# Import mdsuite packages
from mdsuite.analysis.analysis import Analysis

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
plt.style.use('bmh')
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboDiffusionCoefficients(Analysis):
    """ Class for the Green-Kubo diffusion coefficient implementation

    additional attrbs:
        plot
        singular
        distinct
        species
        data_range
    """

    def __init__(self, obj, plot=False, singular=True, distinct=False, species=None, data_range=500, save=True,
                 x_label='Time $(s)$', y_label='VACF $(m^{2}/s^{2})$', analysis_name='Green_Kubo_Diffusion'):
        super().__init__(obj, plot, save, data_range, x_label, y_label, analysis_name)

        self.singular = singular
        self.distinct = distinct
        self.species = species
        self.time = np.linspace(0.0, data_range * self.parent.time_step * self.parent.sample_rate, data_range)
        self.loop_values = np.linspace(0.1 * self.parent.number_of_configurations,
                                       self.parent.number_of_configurations - data_range - 1,
                                       100, dtype=int)
        self.loop_range = len(self.loop_values)

    def _autocorrelation_time(self):
        """ Calculate velocity autocorrelation time

        When performing this analysis, the sampling should occur over the autocorrelation time of the positions in the
        system. This method will calculate what this time is and sample over it to ensure uncorrelated samples.
        """
        pass

    def _singular_diffusion_calculation(self, data_in):
        """ Method to calculate the diffusion coefficients

        Due to the parallelization of our calculations this section of the diffusion coefficient code is seperated
        from the main operation. The method is then called in a loop in the singular_diffusion_coefficients method.
        """

        velocity_matrix = data_in[0]
        item = data_in[1]
        # Calculate the prefactor
        numerator = self.parent.units['length'] ** 2
        denominator = 3 * self.parent.units['time'] * (self.data_range - 1)
        prefactor = numerator / denominator

        coefficient_array = []  # Define the empty coefficient array
        parsed_vacf = np.zeros(self.data_range)  # Instantiate the parsed array

        for i in tqdm(self.loop_values, ncols=70):
            vacf = np.zeros(int(2 * self.data_range - 1))  # Define vacf array
            # Loop over the atoms of species to get the average

            for j in range(len(velocity_matrix)):
                vacf += np.array(
                    signal.correlate(velocity_matrix[j][i:i + self.data_range, 0],
                                     velocity_matrix[j][i:i + self.data_range, 0],
                                     mode='full', method='fft') +
                    signal.correlate(velocity_matrix[j][i:i + self.data_range, 1],
                                     velocity_matrix[j][i:i + self.data_range, 1],
                                     mode='full', method='fft') +
                    signal.correlate(velocity_matrix[j][i:i + self.data_range, 2],
                                     velocity_matrix[j][i:i + self.data_range, 2],
                                     mode='full', method='fft'))

            parsed_vacf += vacf[int(len(vacf) / 2):]  # Update the parsed array

            coefficient_array.append((prefactor / len(velocity_matrix)) * np.trapz(vacf[int(len(vacf) / 2):],
                                                                                   x=self.time))

        # Save data if desired
        if self.save:
            self._save_data(f'{item}_{self.analysis_name}', [self.time, parsed_vacf])

        plt.plot(self.time * self.parent.units['time'], parsed_vacf, label=item)
        if self.plot:
            self._plot_data(title=f'{self.analysis_name}_{item}')

        return [np.mean(coefficient_array), np.std(coefficient_array)/np.sqrt(len(coefficient_array))]

    def _singular_diffusion_coefficients(self):
        """ Calculate the singular diffusion coefficients """

        data = self.parent.load_matrix("Velocities", species=self.species)

        with mp.Pool(processes=4) as p:
            result = p.map(self._singular_diffusion_calculation, zip(data, self.species))

        for i in range(len(self.species)):
            self.parent.diffusion_coefficients["Green-Kubo"]["Singular"][self.species[i]] = result[i]

        if self.plot:
            self._plot_data()

    def _distinct_diffusion_calculation(self, data):
        """ Perform calculation of the distinct coefficients """
        velocity_matrix = data[0]
        tuples = data[1]

        # Define the multiplicative factor
        numerator = self.parent.number_of_atoms * (self.parent.units['length'] ** 2)
        denominator = len(velocity_matrix[tuples[0]]) * len(velocity_matrix[tuples[1]]) * 3 * (
            self.parent.units['time']) * (len(self.time) - 1)
        prefactor = numerator / denominator

        diff_array = []

        plot_array = np.zeros(self.data_range)

        # Loop over reference atoms
        for start in tqdm(self.loop_values, ncols=70):
            vacf = np.zeros(int(2 * self.data_range - 1))  # initialize the vacf array
            for i in range(len(velocity_matrix[tuples[0]])):
                # Loop over test atoms
                for j in range(len(velocity_matrix[tuples[1]])):
                    # Add conditional statement to avoid i=j and alpha=beta
                    if tuples[0] == tuples[1] and j == i:
                        continue

                    vacf += np.array(
                        signal.correlate(velocity_matrix[tuples[0]][i][start:start + self.data_range, 0],
                                         velocity_matrix[tuples[1]][j][start:start + self.data_range:, 0],
                                         mode='full', method='fft') +
                        signal.correlate(velocity_matrix[tuples[0]][i][start:start + self.data_range, 1],
                                         velocity_matrix[tuples[1]][j][start:start + self.data_range, 1],
                                         mode='full', method='fft') +
                        signal.correlate(velocity_matrix[tuples[0]][i][start:start + self.data_range, 2],
                                         velocity_matrix[tuples[1]][j][start:start + self.data_range, 2],
                                         mode='full', method='fft'))

                plot_array += vacf[int(len(vacf) / 2):]

            diff_array.append(prefactor * np.trapz(vacf[int(len(vacf) / 2):], x=self.time))

        plt.plot(self.time, (plot_array / self.loop_range) / abs(min(plot_array / self.loop_range)), label=tuples)
        plt.xlabel(rf'{self.x_label}')  # set the x label
        plt.ylabel(rf'{self.y_label}')  # set the y label
        plt.legend()  # enable the legend
        plt.savefig(f"{self.parent.storage_path}/{self.parent.analysis_name}/Figures/{self.analysis_name}_{tuples}.svg",
                    dpi=600, format='svg')

        if self.save:
            self._save_data(f'{tuples}_{self.analysis_name}', plot_array)

        return [np.mean(diff_array), np.std(diff_array)/np.sqrt(len(diff_array))]

    def _distinct_diffusion_coefficients(self):
        """ Calculate the Green-Kubo distinct diffusion coefficients """

        velocity_matrix = self.parent.load_matrix("Velocities", species=self.species)

        combinations = ['-'.join(tup) for tup in list(itertools.combinations_with_replacement(self.species, 2))]

        index_list = [i for i in range(len(velocity_matrix))]
        index_combinations = list(itertools.combinations_with_replacement(index_list, 2))
        parallel_list = []
        for i in range(len(index_combinations)):
            parallel_list.append([velocity_matrix, index_combinations[i]])

        # Update the dictionary with relevant combinations
        for combination in combinations:
            self.parent.diffusion_coefficients["Green-Kubo"]["Distinct"][combination] = {}

        with mp.Pool(processes=len(index_combinations)) as p:
            result = p.map(self._distinct_diffusion_calculation, parallel_list)

        for i in range(len(combinations)):
            self.parent.diffusion_coefficients["Green-Kubo"]["Distinct"][combinations[i]] = result[i]

    def run_analysis(self):
        """ Run the main analysis """
        self._autocorrelation_time()  # get the correct autocorrelation time

        if self.singular:
            self._singular_diffusion_coefficients()  # calculate the singular diffusion coefficients
        if self.distinct:
            self._distinct_diffusion_coefficients()  # calculate the distinct diffusion coefficients
