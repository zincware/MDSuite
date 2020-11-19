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

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
plt.style.use('bmh')
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class _GreenKuboDiffusionCoefficients:
    """ Class for the Green-Kubo diffusion coefficient implementation

    additional attrbs:
        plot
        singular
        distinct
        species
        data_range
    """

    def __init__(self, obj, plot=False, singular=True, distinct=False, species=None, data_range=500):
        self.parent = obj
        self.plot = plot
        self.singular = singular
        self.distinct = distinct
        self.species = species
        self.data_range = int(data_range)
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

    def _singular_diffusion_coefficients(self):
        """ Calculate the singular diffusion coefficients """

        # Calculate the prefactor
        numerator = self.parent.units['length'] ** 2
        denominator = 3 * self.parent.units['time'] * (len(self.time) - 1)
        prefactor = numerator / denominator

        # Loop over the species in the system
        for item in self.species:
            velocity_matrix = self.parent.load_matrix("Velocities", [item])

            coefficient_array = []  # Define the empty coefficient array
            parsed_vacf = np.zeros(self.data_range)  # Instantiate the parsed array

            for i in tqdm(self.loop_values, ncols=10):
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
            plt.plot(self.time, parsed_vacf, label=item)

            self.parent.diffusion_coefficients["Green-Kubo"]["Singular"][item] = np.mean(coefficient_array)
            plt.plot(self.time, )

        plt.xlabel('Time (ps)')
        plt.ylabel(r'VACF $m^{s}/s^{2}$')
        plt.legend()
        # plt.savefig(f'{self.cwd}/Images/Green_Kubo_Singular_VACF.eps', format='eps')

        if self.plot:
            plt.show()

    def _distinct_diffusion_coefficients(self):
        """ Calculate the Green-Kubo distinct diffusion coefficients """
        print("Please note, distinct diffusion coefficients are not currently accurate")

        velocity_matrix = self.parent.load_matrix("Velocities")

        species = list(self.parent.species.keys())
        combinations = ['-'.join(tup) for tup in list(itertools.combinations_with_replacement(species, 2))]

        index_list = [i for i in range(len(velocity_matrix))]

        # Update the dictionary with relevant combinations
        for combination in combinations:
            self.parent.diffusion_coefficients["Green-Kubo"]["Distinct"][combination] = {}
        pairs = 0
        for tuples in itertools.combinations_with_replacement(index_list, 2):

            # Define the multiplicative factor
            numerator = self.parent.number_of_atoms * (self.parent.units['length'] ** 2)
            denominator = len(velocity_matrix[tuples[0]]) * len(velocity_matrix[tuples[1]]) * 3 * (
                self.parent.units['time']) * (len(self.time) - 1)
            prefactor = numerator / denominator

            diff_array = []

            plot_array = np.zeros(self.data_range)

            # Loop over reference atoms
            for start in tqdm(self.loop_values, ncols=10):
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

            self.parent.diffusion_coefficients["Green-Kubo"]["Distinct"][combinations[pairs]] = \
                [np.mean(diff_array), prefactor * np.std(diff_array) / np.sqrt(len(diff_array))]

            plt.plot(self.time, (plot_array / self.loop_range) / abs(min(plot_array / self.loop_range)), label=tuples)
            pairs += 1

        plt.xlabel("Time (ps)")
        plt.ylabel(r"VACF $m^{2}/s^{2}$")
        plt.legend()
        # plt.savefig("Green_Kubo_Distinct_VACF.eps", format='eps')
        if self.plot:
            plt.show()
