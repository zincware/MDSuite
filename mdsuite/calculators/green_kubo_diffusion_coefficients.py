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
import os
import shutil

# Import user packages
from tqdm import tqdm
import itertools

import tensorflow as tf

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
                 x_label='Time $(s)$', y_label='VACF $(m^{2}/s^{2})$', analysis_name='Green_Kubo_Diffusion',
                 correlation_time=1):
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
        super().__init__(obj, plot, save, data_range, x_label, y_label, analysis_name,
                         correlation_time=correlation_time)

        self.loaded_property = 'Velocities'  # Property to be loaded for the analysis
        self.batch_loop = None  # Number of ensembles in each batch
        self.parallel = False  # Set the parallel attribute
        self.tensor_choice = False  # Load data as a tensor

        self.singular = singular  # calculate the singular coefficients
        self.distinct = distinct  # calculate the distinct coefficients
        self.species = species  # Which species to calculate for

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

    def _collect_data(self):
        """
        Collect data from the tmp directory and return the sum and delete the directory afterwards

        Returns
        -------
        """

        data_array = np.zeros(self.data_range, dtype=float)  # instantiate the array
        files = os.listdir(f"{self.parent.database_path}/tmp")

        for f in files:
            data_array += np.load(f"{self.parent.database_path}/tmp/{f}", allow_pickle=True)

        shutil.rmtree(f"{self.parent.database_path}/tmp")  # delete the directory

        return data_array

    @staticmethod
    @tf.function
    def vectorized_vacf(x, y, z):
        def correlate(data):
            def func(inp):
                return signal.correlate(inp, inp, mode='full', method='auto')

            return tf.py_function(func=func, inp=[data], Tout=tf.float32)

        vacf = tf.map_fn(correlate, x) + \
               tf.map_fn(correlate, y) + \
               tf.map_fn(correlate, z)

        return tf.reduce_sum(vacf, axis=0)

    @staticmethod
    def dataset_map_vacf(data: tf.Tensor) -> tf.Tensor:
        """Compute the velocity autocorrelation function

        Parameters
        ----------
        data: tf.Tensor
            Tensor with the shape (n_atoms, n_timesteps, 3)

        Returns
        -------
        tf.Tensor : reduced sum over the VACF for each timestep

        """

        def correlate(data):
            def func(inp):
                return signal.correlate(inp, inp, mode='full', method='auto')

            return tf.py_function(func=func, inp=[data], Tout=tf.float32)

        vacf = tf.map_fn(correlate, data[:, :, 0]) + \
               tf.map_fn(correlate, data[:, :, 1]) + \
               tf.map_fn(correlate, data[:, :, 2])
        out = tf.reduce_sum(vacf, axis=0)

        return out

    def _singular_diffusion_calculation(self, item: str):
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

        # for i in tqdm(range(int(self.n_batches['Serial'])), ncols=70):
        for i in range(int(self.n_batches['Serial'])):
            batch = self._load_batch(i, item=[item])  # load a batch of data  (n_atoms, timesteps, 3)

            def generator():
                """Generate the data for the VACF Calculation.
                Apply correlation_time and data_range to the data.
                """
                for start_index in range(int(self.batch_loop)):
                    start = start_index + self.correlation_time
                    stop = start + self.data_range
                    yield batch[:, start:stop]

            dataset = tf.data.Dataset.from_generator(
                generator=generator,
                output_signature=tf.TensorSpec(shape=(None, self.data_range, 3), dtype=tf.float32)
            )  # TODO maybe accuracy issues with float32
            dataset = dataset.map(self.dataset_map_vacf, num_parallel_calls=4, deterministic=False)
            # TODO Set num_parallel_calls dynamically
            dataset = dataset.prefetch(64)
            # TODO Set prefetch dynamically

            vacf = tf.zeros(int(self.data_range * 2 - 1))
            for x in tqdm(dataset, total=int(self.batch_loop), desc=f"Processing {item}", smoothing=0.05):
                vacf += x
                parsed_vacf += vacf[int(len(vacf) / 2):]  # Update the parsed array
                coefficient_array.append((prefactor / len(batch)) * np.trapz(vacf[int(len(vacf) / 2):],
                                                                             x=self.time))

        # TODO missing np.save()!
        # Save data if desired
        if self.save:
            self._save_data(f'{item}_{self.analysis_name}', [self.time, parsed_vacf])

        # If plot is needed
        plt.plot(self.time * self.parent.units['time'], parsed_vacf, label=item)
        if self.plot:
            self._plot_data(title=f'{self.analysis_name}_{item}')

        return [str(np.mean(coefficient_array)), str(np.std(coefficient_array) / np.sqrt(len(coefficient_array)))]

    def _singular_diffusion_coefficients(self):
        """
        Calculate the singular diffusion coefficients
        """

        for item in self.species:  # loop over species
            result = self._singular_diffusion_calculation(item=item)  # get the diffusion coefficient
            self._update_properties_file(item='Singular', sub_item=item, data=result)

        # Run the plot data method if needed
        if self.plot:
            self._plot_data()

    def _distinct_correlation_operation(self, start_index, batch, indicator, self_correlation):
        """
        Perform the autocorrelation for the distinct calculation

        Returns
        -------

        """
        vacf = np.zeros(int(2 * self.data_range - 1))  # initialize the vacf array

        if self_correlation:
            a = 0
            b = 0
        else:
            a = 0
            b = 1

        start = start_index + self.correlation_time
        stop = start + self.data_range

        for i in range(len(batch[a])):
            # Loop over test atoms
            for j in range(len(batch[b])):
                # Add conditional statement to avoid i=j and alpha=beta
                if a == b and j == i:
                    continue
                vacf += np.array(
                    signal.correlate(batch[a][i][start:stop, 0],
                                     batch[b][j][start:stop, 0],
                                     mode='full', method='fft') +
                    signal.correlate(batch[a][i][start:stop, 1],
                                     batch[b][j][start:stop, 1],
                                     mode='full', method='fft') +
                    signal.correlate(batch[a][i][start:stop, 2],
                                     batch[b][j][start:stop, 2],
                                     mode='full', method='fft'))

        np.save(f"{self.parent.database_path}/tmp/{indicator}.npy", vacf[int(self.data_range - 1):])
        value = np.trapz(vacf[int(self.data_range - 1):], x=self.time)

        return value

    def _distinct_diffusion_calculation(self, item: list = None):
        """
        Perform calculation of the distinct coefficients
        Currently unavailable
        """

        if item[0] == item[1]:
            molecules = [item[0]]
            atom_factor = len(self.parent.species[molecules[0]]['indices']) * \
                          (len(self.parent.species[molecules[0]]['indices']) - 1)
            self_correlation = True  # tell the code it is computing correlation on the same species

        else:
            molecules = item
            atom_factor = len(self.parent.species[molecules[0]]['indices']) * \
                          (len(self.parent.species[molecules[1]]['indices']))
            self_correlation = False

        numerator = self.parent.units['length'] ** 2
        denominator = 3 * self.parent.units['time'] * (self.data_range - 1) * atom_factor
        prefactor = numerator / denominator

        os.mkdir(f"{self.parent.database_path}/tmp")

        coefficient_array = []
        indicator = 0
        for i in tqdm(range(int(self.n_batches['Parallel'])), ncols=70):
            batch = self._load_batch(i, item=molecules)  # load a batch of data
            if self_correlation:
                batch = [batch]
            for start in range(int(self.batch_loop)):
                vacf = dask.delayed(self._distinct_correlation_operation)(start, batch, indicator, self_correlation)
                coefficient_array.append(prefactor * vacf)
            indicator += 1

            # futures = dask.persist(*coefficient_array)
            if i == 0:
                results = dask.compute(*coefficient_array)
            else:
                results += dask.compute(*coefficient_array)

        parsed_vacf = self._collect_data()  # collect all of the saved files

        # Save data if desired
        if self.save:
            self._save_data(f'{item}_{self.analysis_name}', [self.time, parsed_vacf])

        # If plot is needed
        plt.plot(self.time * self.parent.units['time'], parsed_vacf / max(parsed_vacf), label=item)
        if self.plot:
            self._plot_data(title=f'{self.analysis_name}_{item}')

        return [str(np.mean(results) / (self.n_batches['Serial'])),
                str(np.std(results / (self.n_batches['Serial'])) / np.sqrt(len(results)))]

    def _distinct_diffusion_coefficients(self):
        """
        Calculate the Green-Kubo distinct diffusion coefficients
        """

        self.batch_type = 'Parallel'
        self._calculate_batch_loop()  # update the batch loop for parallel batch sizes
        combinations = ['-'.join(tup) for tup in list(itertools.combinations_with_replacement(self.species, 2))]
        for combination in combinations:
            result = self._distinct_diffusion_calculation(item=combination.split('-'))
            self._update_properties_file(item='Distinct', sub_item=combination, data=result)

    def run_analysis(self):
        """ Run the main analysis """
        self._autocorrelation_time()  # get the correct autocorrelation time
        self._collect_machine_properties()  # collect machine properties and determine batch size
        self._calculate_batch_loop()  # Update the batch loop attribute
        status = self._check_input()  # Check for bad input
        if status == -1:
            return
        else:
            if self.singular:
                self._singular_diffusion_coefficients()  # calculate the singular diffusion coefficients
            if self.distinct:
                self._distinct_diffusion_coefficients()  # calculate the distinct diffusion coefficients
