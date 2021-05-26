"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.
"""

"""
Class for the calculation of the Green-Kubo diffusion coefficients.
Summary
-------
This module contains the code for the Einstein diffusion coefficient class. This class is called by the
Experiment class and instantiated when the user calls the Experiment.einstein_diffusion_coefficients method.
The methods in class can then be called by the Experiment.green_kubo_diffusion_coefficients method and all necessary
calculations performed.
"""

from typing import Union
import numpy as np
import warnings
from tqdm import tqdm
import tensorflow as tf
import itertools
from scipy import signal
import matplotlib.pyplot as plt
from mdsuite.calculators.calculator import Calculator
from mdsuite.utils.meta_functions import join_path

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class EinsteinDistinctDiffusionCoefficients(Calculator):
    """
    Class for the Green-Kubo diffusion coefficient implementation
    Attributes
    ----------
    experiment :  object
            Experiment class to call from
    species : list
            Which species to perform the analysis on
    x_label : str
            X label of the tensor_values when plotted
    y_label : str
            Y label of the tensor_values when plotted
    analysis_name : str
            Name of the analysis
    loaded_property : str
            Property loaded from the database_path for the analysis

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.EinsteinDistinctDiffusionCoefficients(data_range=500, plot=True, correlation_time=10)
    """

    def __init__(self, experiment):
        """
        Constructor for the Green Kubo diffusion coefficients class.

        Attributes
        ----------
        experiment :  object
                Experiment class to call from
        """
        super().__init__(experiment)

        self.scale_function = {'linear': {'scale_factor': 10}}
        self.loaded_property = 'Unwrapped_Positions'  # Property to be loaded for the analysis

        self.database_group = 'Diffusion_Coefficients'
        self.x_label = 'Time $(s)$'
        self.y_label = 'VACF $(m^{2}/s^{2})$'
        self.analysis_name = 'Einstein_Distinct_Diffusion_Coefficients'
        self.experimental = True

        self.msd_array = np.zeros(self.data_range)  # define empty msd array

        if self.species is None:
            self.species = list(self.experiment.species)

        self.combinations = list(itertools.combinations_with_replacement(self.species, 2))

    def __call__(self, plot: bool = False, species: list = None, data_range: int = 500, save: bool = True,
                 correlation_time: int = 1, export: bool = False, atom_selection: dict = np.s_[:], gpu: bool = False):
        """
        Constructor for the Green Kubo diffusion coefficients class.

        Attributes
        ----------
        plot : bool
                if true, plot the tensor_values
        species : list
                Which species to perform the analysis on
        data_range :
                Number of configurations to use in each ensemble
        save :
                If true, tensor_values will be saved after the analysis
        """
        self.update_user_args(plot=plot, data_range=data_range, save=save, correlation_time=correlation_time,
                              atom_selection=atom_selection, export=export, gpu=gpu)

        self.species = species  # Which species to calculate for
        self.msd_array = np.zeros(self.data_range)  # define empty msd array
        self.species = species  # Which species to calculate for
        if self.species is None:
            self.species = list(self.experiment.species)

        self.combinations = list(itertools.combinations_with_replacement(self.species, 2))

        out = self.run_analysis()
        self.experiment.save_class()

        return out

    def _compute_msd(self, data: dict, data_path: list, combination: tuple):
        """
        Compute the vacf on the given dictionary of data.

        Parameters
        ----------
        data : dict
                Dictionary of data returned by tensorflow
        data_path : list
                Data paths for accessing the dictionary
        Returns
        -------
        updates the class state
        """
        for ensemble in tqdm(range(self.ensemble_loop), ncols=70, desc=str(combination)):
            start = ensemble * self.correlation_time
            stop = start + self.data_range
            msd_a = self._msd_operation(data[str.encode(data_path[0])][:, start:stop], square=False)
            msd_b = self._msd_operation(data[str.encode(data_path[0])][:, start:stop], square=False)

            for i in range(len(data[str.encode(data_path[0])])):
                for j in range(i+1, len(data[str.encode(data_path[1])])):
                    if i == j:
                        continue
                    else:
                        self.msd_array += self.prefactor*np.array(tf.reduce_sum(msd_a[i]*msd_b[j], axis=1))

    def run_experimental_analysis(self):
        """
        Perform the distinct coefficient analysis analysis
        """
        if type(self.atom_selection) is dict:
            select_atoms = {}
            for item in self.atom_selection:
                select_atoms[str.encode(join_path(item, "Unwrapped_Positions"))] = self.atom_selection[item]
            self.atom_selection = select_atoms
        for combination in self.combinations:
            type_spec = {}
            self._calculate_prefactor(combination)
            data_path = [join_path(item, 'Unwrapped_Positions') for item in combination]
            self._prepare_managers(data_path=data_path)
            type_spec = self._update_species_type_dict(type_spec, data_path, 3)
            type_spec[str.encode('data_size')] = tf.TensorSpec(None, dtype=tf.int16)
            batch_generator, batch_generator_args = self.data_manager.batch_generator(dictionary=True)
            data_set = tf.data.Dataset.from_generator(batch_generator,
                                                      args=batch_generator_args,
                                                      output_signature=type_spec)
            data_set = data_set.prefetch(tf.data.experimental.AUTOTUNE)
            for batch in data_set:
                self._compute_msd(batch, data_path, combination)
            self._apply_averaging_factor()
            self._post_operation_processes(combination)

    def _calculate_prefactor(self, species: Union[str, tuple] = None):
        """
        calculate the calculator pre-factor.

        Parameters
        ----------
        species : str
                Species property if required.
        Returns
        -------

        """
        if species[0] == species[1]:
            atom_scale = len(self.experiment.species[species[0]]['indices']) * \
                         (len(self.experiment.species[species[1]]['indices']) - 1)
        else:
            atom_scale = len(self.experiment.species[species[0]]['indices']) * \
                         len(self.experiment.species[species[1]]['indices'])
        numerator = self.experiment.units['length'] ** 2
        denominator = 6 * self.experiment.units['time'] * atom_scale
        self.prefactor = numerator / denominator

    def _apply_operation(self, data, index):
        """
        Perform operation on an ensemble.

        Parameters
        ----------
        One tensor_values range of tensor_values to operate on.

        Returns
        -------

        """
        pass

    def _apply_averaging_factor(self):
        """
        Apply an averaging factor to the tensor_values.
        Returns
        -------
        averaged copy of the tensor_values
        """
        self.msd_array /= int(self.n_batches) * self.ensemble_loop

    def _post_operation_processes(self, species: Union[str, tuple] = None):
        """
        call the post-op processes
        Returns
        -------

        """
        if np.sign(self.msd_array[-1]) == -1:
            result = self._fit_einstein_curve([self.time, abs(self.msd_array)])
            properties = {"Property": self.database_group,
                          "Analysis": self.analysis_name,
                          "Subject": "_".join(species),
                          "data_range": self.data_range,
                          'data': -1*result[0],
                          'uncertainty': result[1]}
            self._update_properties_file(properties)
        else:
            result = self._fit_einstein_curve([self.time, self.msd_array])
            properties = {"Property": self.database_group,
                          "Analysis": self.analysis_name,
                          "Subject": species,
                          "data_range": self.data_range,
                          'data': result[0],
                          'uncertainty': result[1]}
            self._update_properties_file(properties)

        # Update the plot if required
        if self.plot:
            plt.plot(np.array(self.time) * self.experiment.units['time'], self.msd_array, label=species)
            plt.show()

        if self.save:
            self._save_data(name=self._build_table_name(species), data=self._build_pandas_dataframe(self.time,
                                                                                                    self.msd_array))
        if self.export:
            self._export_data(name=self._build_table_name(species), data=self._build_pandas_dataframe(self.time,
                                                                                                      self.msd_array))

    def _update_output_signatures(self):
        """
        After having run _prepare managers, update the output signatures.

        Returns
        -------

        """
        pass