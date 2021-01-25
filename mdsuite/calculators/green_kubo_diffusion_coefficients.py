"""
Class for the calculation of the Green-Kubo diffusion coefficients.

Summary
-------
This module contains the code for the Einstein diffusion coefficient class. This class is called by the
Experiment class and instantiated when the user calls the Experiment.einstein_diffusion_coefficients method.
The methods in class can then be called by the Experiment.green_kubo_diffusion_coefficients method and all necessary
calculations performed.
"""

import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import warnings

# Import user packages
from tqdm import tqdm
import itertools

# Import mdsuite packages
from mdsuite.calculators.calculator import Calculator

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
plt.style.use('bmh')
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboDiffusionCoefficients(Calculator):
    """
    Class for the Green-Kubo diffusion coefficient implementation

    Attributes
    ----------
    obj :  object
            Experiment class to call from
    plot : bool
            if true, plot the data
    species : list
            Which species to perform the analysis on
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

    def __init__(self, obj, plot=False, singular=True, distinct=False, species=None, data_range=500, save=True,
                 x_label='Time $(s)$', y_label='VACF $(m^{2}/s^{2})$', analysis_name='Green_Kubo_Diffusion'):
        """
        Python constructor

        Attributes
        ----------
        obj :  object
                Experiment class to call from
        plot : bool
                if true, plot the data
        species : list
                Which species to perform the analysis on
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

        self.loaded_property = 'Velocities'             # Property to be loaded for the analysis
        self.batch_loop = None                          # Number of ensembles in each batch
        self.parallel = False                           # Set the parallel attribute
        self.tensor_choice = False                      # Load data as a tensor

        self.singular = singular                        # calculate the singular coefficients
        self.distinct = distinct                        # calculate the distinct coefficients
        self.species = species                          # Which species to calculate for

        self.database_group = 'diffusion_coefficients'  # Which database group to save the data in

        # Time array
        self.time = np.linspace(0.0, data_range * self.parent.time_step * self.parent.sample_rate, data_range)
        self.correlation_time = 1  # correlation time of the velocities.

        if species is None:
            self.species = list(self.parent.species)

    def _autocorrelation_time(self):
        """
        Calculate velocity autocorrelation time

        When performing this analysis, the sampling should occur over the autocorrelation time of the positions in the
        system. This method will calculate what this time is and sample over it to ensure uncorrelated samples.
        """
        pass

    def _singular_diffusion_calculation(self, item):
        """
        Method to calculate the diffusion coefficients

        Due to the parallelization of our calculations this section of the diffusion coefficient code is separated
        from the main operation. The method is then called in a loop in the singular_diffusion_coefficients method.

        Returns
        -------
        diffusion coeffients : list
                Returns the diffusion coefficient along with the error in its analysis as a list.
        """

        # Calculate the prefactor
        numerator = self.parent.units['length'] ** 2
        denominator = 3 * self.parent.units['time'] * (self.data_range - 1)
        prefactor = numerator / denominator

        coefficient_array = []  # Define the empty coefficient array
        parsed_vacf = np.zeros(self.data_range)  # Instantiate the parsed array

        for i in tqdm(range(int(self.n_batches['Serial'])), ncols=70):
            batch = self._load_batch(i, [item])  # load a batch of data

            for start_index in range(int(self.batch_loop)):
                start = start_index + self.correlation_time
                stop = start + self.data_range

                vacf = np.zeros(int(2 * self.data_range - 1))  # Define vacf array
                # Loop over the atoms of species to get the average
                for j in range(len(batch)):
                    vacf += np.array(
                        signal.correlate(batch[j][start:stop, 0],
                                         batch[j][start:stop, 0],
                                         mode='full', method='auto') +
                        signal.correlate(batch[j][start:stop, 1],
                                         batch[j][start:stop, 1],
                                         mode='full', method='auto') +
                        signal.correlate(batch[j][start:stop, 2],
                                         batch[j][start:stop, 2],
                                         mode='full', method='auto'))

                parsed_vacf += vacf[int(len(vacf) / 2):]  # Update the parsed array

                coefficient_array.append((prefactor / len(batch)) * np.trapz(vacf[int(len(vacf) / 2):],
                                                                                       x=self.time))

        # Save data if desired
        if self.save:
            self._save_data(f'{item}_{self.analysis_name}', [self.time, parsed_vacf])

        # If plot is needed
        plt.plot(self.time * self.parent.units['time'], parsed_vacf, label=item)
        if self.plot:
            self._plot_data(title=f'{self.analysis_name}_{item}')

        return [np.mean(coefficient_array), np.std(coefficient_array)/np.sqrt(len(coefficient_array))]

    def _singular_diffusion_coefficients(self):
        """
        Calculate the singular diffusion coefficients
        """

        for item in self.species:                                                        # loop over species
            result = self._singular_diffusion_calculation(item=item)                     # get the diffusion coefficient
            self.parent.diffusion_coefficients["Green-Kubo"]["Singular"][item] = result  # Update the class

        # Run the plot data method if needed
        if self.plot:
            self._plot_data()

    def _distinct_diffusion_calculation(self, data):
        """
        Perform calculation of the distinct coefficients

        Currently unavailable
        """
        """
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
        for start in tqdm(50, ncols=70):
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

        plt.plot(self.time, (plot_array) / abs(min(plot_array / self.loop_range)), label=tuples)
        plt.xlabel(rf'{self.x_label}')  # set the x label
        plt.ylabel(rf'{self.y_label}')  # set the y label
        plt.legend()  # enable the legend
        plt.savefig(f"{self.parent.storage_path}/{self.parent.analysis_name}/Figures/{self.analysis_name}_{tuples}.svg",
                    dpi=600, format='svg')

        if self.save:
            self._save_data(f'{tuples}_{self.analysis_name}', plot_array)

        return [np.mean(diff_array), np.std(diff_array)/np.sqrt(len(diff_array))]
        """
        raise NotImplementedError

    def _distinct_diffusion_coefficients(self):
        """
        Calculate the Green-Kubo distinct diffusion coefficients
        """
        """
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
        """
        raise NotImplementedError

    def run_analysis(self):
        """ Run the main analysis """
        self._autocorrelation_time()                    # get the correct autocorrelation time
        self._collect_machine_properties()              # collect machine properties and determine batch size
        self._calculate_batch_loop()                    # Update the batch loop attribute
        status = self._check_input()                    # Check for bad input
        if status == 0:
            return
        else:
            if self.singular:
                self._singular_diffusion_coefficients()     # calculate the singular diffusion coefficients
            if self.distinct:
                self._distinct_diffusion_coefficients()     # calculate the distinct diffusion coefficients
