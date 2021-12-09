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
import itertools
from typing import Union

import numpy as np
import tensorflow as tf
from bokeh.models import Span
from scipy import signal
from tqdm import tqdm

from mdsuite.calculators.calculator import Calculator, call
from mdsuite.utils.meta_functions import join_path


class GreenKuboDistinctDiffusionCoefficients(Calculator):
    """
    Class for the Green-Kubo diffusion coefficient implementation
    Attributes
    ----------
    experiment :  object
            Experiment class to call from
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
    experiment.run_computation.GreenKuboDistinctDiffusionCoefficients(data_range=500,
    plot=True, correlation_time=10)
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

        self.scale_function = {"linear": {"scale_factor": 5}}
        self.loaded_property = "Velocities"

        self.database_group = "Diffusion_Coefficients"
        self.x_label = r"$$\text{Time} / s$$"
        self.y_label = r"\text{VACF}  / m^{2}/s^{2}$$"
        self.analysis_name = "Green_Kubo_Distinct_Diffusion_Coefficients"
        self.experimental = True

    @call
    def __call__(
        self,
        plot: bool = False,
        species: list = None,
        data_range: int = 500,
        save: bool = True,
        correlation_time: int = 1,
        export: bool = False,
        atom_selection: dict = np.s_[:],
        gpu: bool = False,
        integration_range: int = None,
    ):
        """
        Constructor for the Green Kubo diffusion coefficients class.

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
        integration_range : int
                Range over which to perform the integration.
        """
        self.update_user_args(
            plot=plot,
            data_range=data_range,
            save=save,
            correlation_time=correlation_time,
            atom_selection=atom_selection,
            export=export,
            gpu=gpu,
        )

        self.species = species  # Which species to calculate for

        self._return_arrays = {}

        self.vacf = np.zeros(self.data_range)
        self.sigma = []

        if integration_range is None:
            self.integration_range = self.data_range
        else:
            self.integration_range = integration_range

        if self.species is None:
            self.species = list(self.experiment.species)

        self.combinations = list(
            itertools.combinations_with_replacement(self.species, 2)
        )

    def _compute_vacf(self, data: dict, data_path: list, combination: tuple):
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
        for ensemble in tqdm(
            range(self.ensemble_loop), ncols=70, desc=str(combination)
        ):
            self.vacf = np.zeros(self.data_range)
            start = ensemble * self.correlation_time
            stop = start + self.data_range
            vacf = np.zeros(self.data_range)
            for i in range(len(data[str.encode(data_path[0])])):
                for j in range(i + 1, len(data[str.encode(data_path[1])])):
                    if i == j:
                        continue
                    else:
                        vacf += sum(
                            [
                                signal.correlate(
                                    data[str.encode(data_path[0])][i][start:stop, idx],
                                    data[str.encode(data_path[1])][j][start:stop, idx],
                                    mode="full",
                                    method="auto",
                                )
                                for idx in range(3)
                            ]
                        )
            self.vacf += vacf[
                int(self.data_range - 1) :
            ]  # Update the averaged function
            self.sigma.append(np.trapz(vacf[int(self.data_range - 1) :], x=self.time))

    def run_experimental_analysis(self):
        """
        Perform the distinct coefficient analysis analysis
        """
        if type(self.atom_selection) is dict:
            select_atoms = {}
            for item in self.atom_selection:
                select_atoms[
                    str.encode(join_path(item, "Velocities"))
                ] = self.atom_selection[item]
            self.atom_selection = select_atoms
        for combination in self.combinations:
            type_spec = {}
            data_path = [join_path(item, "Velocities") for item in combination]
            self._prepare_managers(data_path=data_path)
            type_spec = self._update_species_type_dict(type_spec, data_path, 3)
            type_spec[str.encode("data_size")] = tf.TensorSpec(None, dtype=self.dtype)
            batch_generator, batch_generator_args = self.data_manager.batch_generator(
                dictionary=True
            )
            data_set = tf.data.Dataset.from_generator(
                batch_generator, args=batch_generator_args, output_signature=type_spec
            )
            data_set = data_set.prefetch(tf.data.experimental.AUTOTUNE)
            for batch in data_set:
                self._compute_vacf(batch, data_path, combination)
            self._calculate_prefactor(combination)
            self._apply_averaging_factor()
            self._post_operation_processes(combination)
            self._return_arrays[str(combination)] = self.vacf
        return self._return_arrays

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
            atom_scale = self.experiment.species[species[0]].n_particles * (
                self.experiment.species[species[1]].n_particles - 1
            )
        else:
            atom_scale = (
                self.experiment.species[species[0]].n_particles
                * self.experiment.species[species[1]].n_particles
            )
        numerator = self.experiment.units["length"] ** 2
        denominator = (
            3 * self.experiment.units["time"] * (self.data_range - 1) * atom_scale
        )

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
        pass

    def _post_operation_processes(self, species: Union[str, tuple] = None):
        """
        call the post-op processes
        Returns
        -------

        """
        result = self.prefactor * np.array(self.sigma)

        data = {
            "diffusion_coefficient": np.mean(result).tolist(),
            "uncertainty": (np.std(result) / (np.sqrt(len(result)))).tolist(),
            "time": self.time.tolist(),
            "acf": self.vacf.tolist(),
        }

        self.queue_data(data=data, subjects=list(species))

        # Update the plot if required
        if self.plot:
            span = Span(
                location=(np.array(self.time) * self.experiment.units["time"])[
                    self.integration_range - 1
                ],
                dimension="height",
                line_dash="dashed",
            )
            self.run_visualization(
                x_data=np.array(self.time) * self.experiment.units["time"],
                y_data=self.vacf.numpy(),
                title=(
                    f"{species}: {np.mean(result): .3E} +-"
                    f" {np.std(result) / (np.sqrt(len(result))): .3E}"
                ),
                layouts=[span],
            )

    def _update_output_signatures(self):
        """
        After having run _prepare managers, update the output signatures.

        Returns
        -------

        """
        pass
