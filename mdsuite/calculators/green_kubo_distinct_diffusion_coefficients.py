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

import tensorflow as tf

# Import mdsuite packages
from mdsuite.calculators.calculator import Calculator

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboDistinctDiffusionCoefficients(Calculator):
    """
    Class for the Green-Kubo diffusion coefficient implementation
    Attributes
    ----------
    experiment :  object
            Experiment class to call from
    plot : bool
            if true, plot the tensor_values
    species : list
            Which species to perform the analysis on
    data_range :
            Number of configurations to use in each ensemble
    save :
            If true, tensor_values will be saved after the analysis
    x_label : str
            X label of the tensor_values when plotted
    y_label : str
            Y label of the tensor_values when plotted
    analysis_name : str
            Name of the analysis
    loaded_property : str
            Property loaded from the database_path for the analysis
    correlation_time : int
            Correlation time of the property being studied. This is used to ensure ensemble sampling is only performed
            on uncorrelated samples. If this is true, the error extracted form the calculation will be correct.
    """

    def __init__(self, experiment, species: list = None, data_range: int = 500, correlation_time: int = 1, **kwargs):
        """
        Constructor for the Green Kubo diffusion coefficients class.

        Attributes
        ----------
        experiment :  object
                Experiment class to call from
        plot : bool
                if true, plot the tensor_values
        species : list
                Which species to perform the analysis on
        data_range :
                Number of configurations to use in each ensemble
        save :
                If true, tensor_values will be saved after the analysis
        """
        super().__init__(experiment, data_range=data_range, correlation_time=correlation_time, **kwargs)

        self.loaded_property = 'Velocities'  # Property to be loaded for the analysis
        self.species = species  # Which species to calculate for

        self.database_group = 'diffusion_coefficients'  # Which database_path group to save the tensor_values in
        self.x_label = 'Time $(s)$'
        self.y_label = 'VACF $(m^{2}/s^{2})$'
        self.analysis_name = 'Green_Kubo_Diffusion'

        self.vacf = np.zeros(self.data_range)
        self.sigma = []

        if self.species is None:
            self.species = list(self.experiment.species)

    def _distinct_diffusion_calculation(self, item: list = None):
        """
        Perform calculation of the distinct coefficients
        Currently unavailable
        """

        if item[0] == item[1]:
            molecules = [item[0]]
            atom_factor = len(self.experiment.species[molecules[0]]['indices']) * \
                          (len(self.experiment.species[molecules[0]]['indices']) - 1)
            self_correlation = True  # tell the code it is computing correlation on the same species

        else:
            molecules = item
            atom_factor = len(self.experiment.species[molecules[0]]['indices']) * \
                          (len(self.experiment.species[molecules[1]]['indices']))
            self_correlation = False

        numerator = self.experiment.units['length'] ** 2
        denominator = 3 * self.experiment.units['time'] * (self.data_range - 1) * atom_factor
        prefactor = numerator / denominator

        for i in range(int(self.n_batches['Parallel'])):
            print("Start Loading Data")
            batch = self.load_batch(i, item=molecules)  # load a batch of tensor_values
            print('Done!')
            a = 0
            if self_correlation:
                batch = [batch]
                b = 0
                ab_diagonal_length = len(batch[a])  # for TQDM only
            else:
                b = 1
                ab_diagonal_length = 0  # for TQDM only

            def generator():
                for start in range(int(self.batch_loop)):
                    stop = start + self.data_range
                    for i in range(len(batch[a])):
                        for j in range(len(batch[b])):
                            if a == b and j == i:
                                continue
                            yield batch[a][i][start:stop], batch[b][j][start:stop]

            dataset = tf.data.Dataset.from_generator(
                generator=generator,
                output_signature=(
                    tf.TensorSpec(shape=(self.data_range, 3), dtype=tf.float64),
                    tf.TensorSpec(shape=(self.data_range, 3), dtype=tf.float64)
                )
            )

            dataset = dataset.map(self._convolution_op, num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                  deterministic=False)
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)  # Doesn't seem to be much of an improvement

            parsed_vacf = tf.zeros(int(self.data_range), dtype=tf.float64)
            vacf = tf.zeros(int(self.data_range * 2 - 1), dtype=tf.float64)
            for x in tqdm(dataset.batch(self.data_range),
                          total=int(self.batch_loop * len(batch[a] * len(batch[b] - ab_diagonal_length))),
                          desc=f"Processing {item}", smoothing=0.05):
                vacf += tf.reduce_sum(x, axis=0)
                vacf *= prefactor
                parsed_vacf += vacf[int(len(vacf) / 2):]

                # TODO where to save the VACFs?

            plt.plot(self.time * self.experiment.units['time'], parsed_vacf / max(parsed_vacf), label=item)

        results = vacf

        # Save tensor_values if desired
        if self.save:
            self._save_data(f'{item}_{self.analysis_name}', [self.time, parsed_vacf])

        # If plot is needed
        plt.plot(self.time * self.experiment.units['time'], parsed_vacf / max(parsed_vacf), label=item)
        if self.plot:
            self._plot_data(title=f'{self.analysis_name}_{item}')

        return [str(np.mean(results) / (self.n_batches['Serial'])),
                str(np.std(results / (self.n_batches['Serial'])) / np.sqrt(len(results)))]

    def _distinct_diffusion_coefficients(self):
        """
        Calculate the Green-Kubo distinct diffusion coefficients
        """

        combinations = ['-'.join(tup) for tup in list(itertools.combinations_with_replacement(self.species, 2))]
        for combination in combinations:
            result = self._distinct_diffusion_calculation(item=combination.split('-'))
            self._update_properties_file(item='Distinct', sub_item=combination, data=result)