"""
MDSuite: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
"""
from __future__ import annotations
import logging
import numpy as np
import warnings
from typing import Union, Any, List
from tqdm import tqdm
import tensorflow as tf
from mdsuite.calculators.calculator import Calculator, call

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdsuite.database.scheme import Computation

tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)


class EinsteinDiffusionCoefficients(Calculator):
    """
    Class for the Einstein diffusion coefficient implementation

    Description:
    This module contains the code for the Einstein diffusion coefficient class.
    This class is called by the Experiment class and instantiated when the user calls
    the  Experiment.einstein_diffusion_coefficients method. The methods in class can
    then be called by the Experiment.einstein_diffusion_coefficients method and all
    necessary calculations performed.

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
    experiment.run_computation.EinsteinDiffusionCoefficients(data_range=500,
                                                             plot=True,
                                                             correlation_time=10)
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
        self.scale_function = {"linear": {"scale_factor": 150}}
        self.loaded_property = "Unwrapped_Positions"
        self.species = None
        self.molecules = None
        self.database_group = "Diffusion_Coefficients"
        self.x_label = r"$ \text{Time} / s$"
        self.y_label = r"$ \text{MSD} / m^{2}$"
        self.result_keys = ["diffusion_coefficient", "uncertainty"]
        self.result_series_keys = ["time", "msd"]
        self.analysis_name = "Einstein Self-Diffusion Coefficients"
        self.loop_condition = False
        self.optimize = None
        self.msd_array = None  # define empty msd array
        self.tau_values = None
        self.species = list()
        log.info("starting Einstein Diffusion Computation")

    @call
    def __call__(
        self,
        plot: bool = True,
        species: list = None,
        data_range: int = 100,
        save: bool = True,
        optimize: bool = False,
        correlation_time: int = 1,
        atom_selection: np.s_ = np.s_[:],
        molecules: bool = False,
        tau_values: Union[int, List, Any] = np.s_[:],
        gpu: bool = False,
    ) -> None:

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
        self.update_user_args(
            plot=plot,
            data_range=data_range,
            save=save,
            correlation_time=correlation_time,
            atom_selection=atom_selection,
            tau_values=tau_values,
            gpu=gpu,
        )

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

        return self.update_db_entry_with_kwargs(
            data_range=data_range,
            correlation_time=correlation_time,
            molecules=molecules,
            atom_selection=atom_selection,
            tau_values=tau_values,
            species=species,
        )

    def _update_output_signatures(self):
        """
        After having run _prepare managers, update the output signatures.

        Returns
        -------
        Update the class state.
        """
        self.batch_output_signature = tf.TensorSpec(
            shape=(None, self.batch_size, 3), dtype=tf.float64
        )
        self.ensemble_output_signature = tf.TensorSpec(
            shape=(None, self.data_range, 3), dtype=tf.float64
        )

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
            numerator = self.experiment.units["length"] ** 2
            denominator = (
                self.experiment.units["time"]
                * len(self.experiment.molecules[species]["indices"])
            ) * 6
        else:
            # Calculate the prefactor
            numerator = self.experiment.units["length"] ** 2
            denominator = (
                self.experiment.units["time"]
                * len(self.experiment.species[species]["indices"])
            ) * 6

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

        data = {
            self.result_keys[0]: result[0],
            self.result_keys[1]: result[1],
            self.result_series_keys[0]: self.time.tolist(),
            self.result_series_keys[1]: self.msd_array.tolist(),
        }

        self.queue_data(data=data, subjects=[species])
