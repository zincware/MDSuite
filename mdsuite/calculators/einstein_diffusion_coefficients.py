"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Class for the calculation of the einstein diffusion coefficients.

Summary
-------
Description: This module contains the code for the Einstein diffusion coefficient class. This class is called by the
Experiment class and instantiated when the user calls the Experiment.einstein_diffusion_coefficients method.
The methods in class can then be called by the Experiment.einstein_diffusion_coefficients method and all necessary
calculations performed.
"""
from __future__ import annotations
import logging
import matplotlib.pyplot as plt
import numpy as np
import warnings
from typing import Union, Any, List
from tqdm import tqdm
import tensorflow as tf
from mdsuite.calculators.calculator import Calculator, call
from mdsuite.database.calculator_database import Parameters

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdsuite.experiment import Experiment

tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)


class EinsteinDiffusionCoefficients(Calculator):
    """
    Class for the Einstein diffusion coefficient implementation

    Description: This module contains the code for the Einstein diffusion coefficient class.
    This class is called by the Experiment class and instantiated when the user calls the
    Experiment.einstein_diffusion_coefficients method. The methods in class can then be
    called by the Experiment.einstein_diffusion_coefficients method and all necessary
    calculations performed.

    Attributes
    ----------
    experiment :  Experiment
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
    experiment.run_computation.EinsteinDiffusionCoefficients(data_range=500, plot=True, correlation_time=10)
    """

    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        experiment :  Experiment
                Experiment class to call from
        experiments :  Experiment
                Experiment classes to call from
        load_data :  bool
                whether to load data or not
        """

        super().__init__(**kwargs)
        self.scale_function = {'linear': {'scale_factor': 150}}
        self.loaded_property = 'Unwrapped_Positions'
        self.species = None
        self.molecules = None
        self.database_group = 'Diffusion_Coefficients'
        self.x_label = 'Time (s)'
        self.y_label = 'MSD (m$^2$)'
        self.analysis_name = 'Einstein_Self_Diffusion_Coefficients'
        self.loop_condition = False
        self.optimize = None
        self.msd_array = None  # define empty msd array
        self.tau_values = None
        self.species = list()
        log.info('starting Einstein Diffusion Computation')

    @call
    def __call__(self, plot: bool = True,
                 species: list = None,
                 data_range: int = 100,
                 save: bool = True,
                 optimize: bool = False,
                 correlation_time: int = 1,
                 atom_selection: np.s_ = np.s_[:],
                 export: bool = False,
                 molecules: bool = False,
                 tau_values: Union[int, List, Any] = np.s_[:],
                 gpu: bool = False):

        """

        Parameters
        ----------
        plot : bool
                if true, plot the output.
        species : list
                List of species on which to operate.
        data_range : int
                Data range to use in the analysis.
        save : bool
                if true, save the output.
        optimize : bool
                If true, an optimization loop will be run.
        correlation_time : int
                Correlation time to use in the window sampling.
        atom_selection : np.s_
                Selection of atoms to use within the HDF5 database.
        export : bool
                If true, export the data directly into a csv file.
        molecules : bool
                If true, molecules are used instead of atoms.
        tau_values : Union[int, list, np.s_]
                Selection of tau values to use in the window sliding.
        gpu : bool
                If true, scale the memory requirement down to the amount of
                the biggest GPU in the system.

        Returns
        -------
        None
        """

        self.update_db_entry_with_kwargs(
            data_range=data_range,
            correlation_time=correlation_time,
            atom_selection=atom_selection,
            tau_values=tau_values,
        )

        self.update_user_args(plot=plot,
                              data_range=data_range,
                              save=save,
                              correlation_time=correlation_time,
                              atom_selection=atom_selection,
                              tau_values=tau_values,
                              export=export,
                              gpu=gpu)
        self.species = species
        self.molecules = molecules
        self.optimize = optimize
        # attributes based on user args
        self.msd_array = np.zeros(self.data_resolution)  # define empty msd array

        if species is None:
            if molecules:
                self.species = list(self.experiment.molecules)
            else:
                self.species = list(self.experiment.species)

    def _update_output_signatures(self):
        """
        After having run _prepare managers, update the output signatures.

        Returns
        -------
        Update the class state.
        """
        self.batch_output_signature = tf.TensorSpec(shape=(None, self.batch_size, 3),
                                                    dtype=tf.float64)
        self.ensemble_output_signature = tf.TensorSpec(shape=(None, self.data_range, 3),
                                                       dtype=tf.float64)

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
        if self.molecules:
            # Calculate the prefactor
            numerator = self.experiment.units['length'] ** 2
            denominator = (self.experiment.units['time'] * len(self.experiment.molecules[species]['indices'])) * 6
        else:
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
        msd = self._msd_operation(ensemble)

        # Sum over trajectory and then coordinates and apply averaging and pre-factors
        msd = self.prefactor * tf.reduce_sum(tf.reduce_sum(msd, axis=0), axis=1)
        self.msd_array += np.array(msd)  # Update the averaged function

    def _post_operation_processes(self, species: str = None):
        """
        Apply post-op processes such as saving and plotting.

        Returns
        -------

        """

        result = self._fit_einstein_curve([self.time, self.msd_array])
        log.debug(f"Saving {species}")
        properties = Parameters(
            Property=self.database_group,
            Analysis=self.analysis_name,
            data_range=self.data_range,
            data=[{'x': result[0], 'uncertainty': result[1]}],
            Subject=[species]
        )

        if self.save:
            data = properties.data
            data += [{'time': x, 'msd': y} for x, y in zip(self.time, self.msd_array)]
            properties.data = data

        self.update_database(properties)

        if self.export:
            self._export_data(name=self._build_table_name(species), data=self._build_pandas_dataframe(self.time,
                                                                                                      self.msd_array))

        if self.plot:
            plt.xlabel(rf'{self.x_label}')  # set the x label
            plt.ylabel(rf'{self.y_label}')  # set the y label
            plt.plot(np.array(self.time) * self.experiment.units['time'],
                     self.msd_array * self.experiment.units['time'],
                     label=fr"{species}: {result[0]: 0.3E} $\pm$ {result[1]: 0.3E}")
