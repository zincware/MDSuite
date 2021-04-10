"""
Class for the calculation of the einstein diffusion coefficients.

Summary
-------
Description: This module contains the code for the Einstein diffusion coefficient class. This class is called by the
Experiment class and instantiated when the user calls the Experiment.einstein_diffusion_coefficients method.
The methods in class can then be called by the Experiment.einstein_diffusion_coefficients method and all necessary
calculations performed.
"""

import logging

# Python standard packages
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Import user packages
from tqdm import tqdm
import tensorflow as tf

# Import MDSuite packages
from mdsuite.calculators.calculator import Calculator

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class EinsteinDiffusionCoefficients(Calculator):
    """
    Class for the Einstein diffusion coefficient implementation

    Description: This module contains the code for the Einstein diffusion coefficient class. This class is called by the
    Experiment class and instantiated when the user calls the Experiment.einstein_diffusion_coefficients method.
    The methods in class can then be called by the Experiment.einstein_diffusion_coefficients method and all necessary
    calculations performed.

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

    def __init__(self, experiment, plot: bool = True, species: list = None, data_range: int = 100, save: bool = True,
                 optimize: bool = False, correlation_time: int = 1, atom_selection=np.s_[:]):
        """

        Parameters
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
        optimize : bool
                If true, the tensor_values range will be optimized
        """

        super().__init__(experiment, plot, save, data_range, correlation_time=correlation_time,
                         atom_selection=atom_selection)
        self.scale_function = {'linear': {'scale_factor': 5}}

        self.loaded_property = 'Unwrapped_Positions'    # Property to be loaded

        self.species = species                          # Which species to calculate the diffusion for

        self.database_group = 'diffusion_coefficients'
        self.x_label = 'Time (s)'
        self.y_label = 'MSD (m$^2$/s)'
        self.analysis_name = 'einstein_diffusion_coefficients'

        self.loop_condition = False                     # Condition used when tensor_values range optimizing
        self.optimize = optimize                        # optimize the tensor_values range

        self.msd_array = np.zeros(self.data_range)  # define empty msd array

        if species is None:
            self.species = list(self.experiment.species)

        self.log = logging.getLogger(__name__)

        self.log.info('starting Einstein Diffusion Computation')

    def _update_output_signatures(self):
        """
        After having run _prepare managers, update the output signatures.

        Returns
        -------
        Update the class state.
        """
        self.batch_output_signature = tf.TensorSpec(shape=(None, self.batch_size, 3), dtype=tf.float64)
        self.ensemble_output_signature = tf.TensorSpec(shape=(None, self.data_range, 3), dtype=tf.float64)

    def _calculate_prefactor(self, species: str = None):
        """
        Compute the prefactor

        Parameters
        ----------
        species : str
                Species being studied.

        Returns
        -------
        Updates the class state.
        """
        # Calculate the prefactor
        numerator = self.experiment.units['length'] ** 2
        denominator = (self.experiment.units['time'] * len(self.experiment.species[species]['indices'])) * 6
        self.prefactor = numerator / denominator

    def _apply_averaging_factor(self):
        """
        Apply the averaging factor to the msd array.
        Returns
        -------

        """
        self.msd_array /= int(self.n_batches) * self.ensemble_loop

    def _apply_operation(self, ensemble, index):
        """
        Calculate and return the msd.

        Parameters
        ----------
        ensemble

        Returns
        -------
        MSD of the tensor_values.
        """
        msd = (ensemble - (
            tf.repeat(tf.expand_dims(ensemble[:, 0], 1), self.data_range, axis=1))) ** 2

        # Sum over trajectory and then coordinates and apply averaging and prefactors
        msd = self.prefactor * tf.reduce_sum(tf.reduce_sum(msd, axis=0), axis=1)
        self.msd_array += np.array(msd)  # Update the averaged function

    def _post_operation_processes(self, species: str = None):
        """
        Apply post-op processes such as saving and plotting.
        Returns
        -------

        """

        result = self._fit_einstein_curve([self.time, self.msd_array])
        self._update_properties_file(item='Singular', sub_item=species, data=result)
        # Update the plot if required
        if self.plot:
            plt.plot(np.array(self.time) * self.experiment.units['time'], self.msd_array, label=species)

        # Save the array if required
        if self.save:
            self._save_data(f"{species}_{self.analysis_name}", [self.time, self.msd_array])

    def _optimized_calculation(self):
        """
        Run an range optimized calculation
        """
        # Optimize the data_range parameter
        # for item in self.species:
        #     while not self.loop_condition:
        #         tensor_values = self._self_diffusion_coefficients(item, parse=True)
        #         self._optimize_einstein_data_range(tensor_values=tensor_values)
        #
        #     self.loop_condition = False
        #     result = self._fit_einstein_curve(tensor_values)  # get the final fits
        #     self._update_properties_file(item='Singular', sub_item=item, tensor_values=result)
        pass
