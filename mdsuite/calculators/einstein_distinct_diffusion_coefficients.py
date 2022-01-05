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
Module for computing distinct diffusion coefficients using the Einstein method.
"""
import itertools
import warnings
from dataclasses import dataclass
from typing import Any, List, Union

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from mdsuite.calculators.calculator import call
from mdsuite.calculators.trajectory_calculator import TrajectoryCalculator
from mdsuite.database import simulation_properties
from mdsuite.utils.meta_functions import join_path


@dataclass
class Args:
    """
    Data class for the saved properties.
    """

    data_range: int
    correlation_time: int
    atom_selection: np.s_
    tau_values: np.s_
    molecules: bool
    species: list
    integration_range: int


tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class EinsteinDistinctDiffusionCoefficients(TrajectoryCalculator):
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
    experiment.run_computation.EinsteinDistinctDiffusionCoefficients(data_range=500,
                                                                     plot=True,
                                                                     correlation_time=10)
    """

    def __init__(self, **kwargs):
        """
        Constructor for the Green Kubo diffusion coefficients class.

        Attributes
        ----------
        experiment :  object
                Experiment class to call from
        """
        super().__init__(**kwargs)

        self.scale_function = {"linear": {"scale_factor": 10}}
        self.loaded_property = simulation_properties.unwrapped_positions

        self.database_group = "Diffusion_Coefficients"
        self.x_label = r"$$\text{Time} / s $$"
        self.y_label = r"$$\text{VACF} / m^{2}/s^{2}$$"
        self.analysis_name = "Einstein_Distinct_Diffusion_Coefficients"
        self.experimental = True

        self.combinations = []

    @call
    def __call__(
        self,
        plot: bool = False,
        species: list = None,
        data_range: int = 500,
        save: bool = True,
        correlation_time: int = 1,
        tau_values: Union[int, List, Any] = np.s_[:],
        molecules: bool = False,
        export: bool = False,
        atom_selection: dict = np.s_[:],
        gpu: bool = False,
    ):
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

        correlation_time : int
                Correlation time to use in the window sampling.
        atom_selection : np.s_
                Selection of atoms to use within the HDF5 database.
        export : bool
                If true, export the data directly into a csv file.
        gpu : bool
                If true, scale the memory requirement down to the amount of
                the biggest GPU in the system.

        Returns
        -------
        None

        """

        if species is None:
            species = list(self.experiment.species)
        self.combinations = list(itertools.combinations_with_replacement(species, 2))

        # set args that will affect the computation result
        self.args = Args(
            data_range=data_range,
            correlation_time=correlation_time,
            atom_selection=atom_selection,
            tau_values=tau_values,
            molecules=molecules,
            species=species,
        )

        self.msd_array = np.zeros(self.args.data_range)  # define empty msd array

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
        msd_a = self._msd_operation(data[str.encode(data_path[0])], square=False)
        msd_b = self._msd_operation(data[str.encode(data_path[0])], square=False)

        for i in range(len(data[str.encode(data_path[0])])):
            for j in range(i + 1, len(data[str.encode(data_path[1])])):
                if i == j:
                    continue
                else:
                    self.msd_array += self.prefactor * np.array(
                        tf.reduce_sum(msd_a[i] * msd_b[j], axis=1)
                    )

    def run_experimental_analysis(self):
        """
        Perform the distinct coefficient analysis analysis
        """
        if type(self.atom_selection) is dict:
            select_atoms = {}
            for item in self.atom_selection:
                select_atoms[
                    str.encode(join_path(item, "Unwrapped_Positions"))
                ] = self.atom_selection[item]
            self.atom_selection = select_atoms
        for combination in self.combinations:
            type_spec = {}
            self._calculate_prefactor(combination)
            data_path = [join_path(item, "Unwrapped_Positions") for item in combination]
            self._prepare_managers(data_path=data_path)
            type_spec = self._update_species_type_dict(type_spec, data_path, 3)
            type_spec[str.encode("data_size")] = tf.TensorSpec(None, dtype=tf.int16)
            batch_generator, batch_generator_args = self.data_manager.batch_generator(
                dictionary=True
            )
            data_set = tf.data.Dataset.from_generator(
                batch_generator, args=batch_generator_args, output_signature=type_spec
            )
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
            atom_scale = len(self.experiment.species[species[0]]["indices"]) * (
                len(self.experiment.species[species[1]]["indices"]) - 1
            )
        else:
            atom_scale = len(self.experiment.species[species[0]]["indices"]) * len(
                self.experiment.species[species[1]]["indices"]
            )
        numerator = self.experiment.units["length"] ** 2
        denominator = 6 * self.experiment.units["time"] * atom_scale
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

            data = {"diffusion_coefficient": -1 * result[0], "uncertainty": result[1]}
        else:
            result = self._fit_einstein_curve([self.time, self.msd_array])
            data = {"diffusion_coefficient": result[0], "uncertainty": result[1]}

        data.update({"time": self.time.tolist(), "msd": self.msd_array.tolist()})

        self.queue_data(data=data, subjects=list(species))

        # Update the plot if required
        if self.plot:
            self.run_visualization(
                x_data=np.array(self.time) * self.experiment.units["time"],
                y_data=self.msd_array * self.experiment.units["time"],
                title=species,
            )

    def _update_output_signatures(self):
        """
        After having run _prepare managers, update the output signatures.

        Returns
        -------

        """
        pass

    def run_calculator(self):
        """
        Perform the distinct coefficient analysis analysis
        """
        self.check_input()
        for combination in self.combinations:
            species_values = list(combination)
            dict_ref = [
                str.encode("/".join([species, self.loaded_property[0]]))
                for species in species_values
            ]
            batch_ds = self.get_batch_dataset(species_values)

            for batch in batch_ds:
                ensemble_ds = self.get_ensemble_dataset(batch, species_values)
                for ensemble in ensemble_ds:
                    self.ensemble_operation(ensemble, dict_ref)

            self._calculate_prefactor(combination)
            self._post_operation_processes(combination)
            self._return_arrays[str(combination)] = self.vacf

        return self._return_arrays
