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
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
from bokeh.models import Span
from tqdm import tqdm
from typing import Union
from mdsuite.calculators.calculator import Calculator, call
from mdsuite.database.scheme import Computation
from dataclasses import dataclass
from mdsuite.database import simulation_properties
from typing import List, Any


@dataclass
class Args:
    data_range: int
    correlation_time: int
    atom_selection: np.s_
    tau_values: np.s_
    molecules: bool
    species: list
    integration_range: int


class GreenKuboDiffusionCoefficients(Calculator):
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
    experiment.run_computation.GreenKuboSelfDiffusionCoefficients(data_range=500,
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

        self.loaded_property = simulation_properties.velocities
        self.scale_function = {"linear": {"scale_factor": 150}}

        self.database_group = "Diffusion_Coefficients"
        self.x_label = r"$$\text{Time} / s$$"
        self.y_label = r"$$\text{VACF} / m^{2}/s^{2}$$"
        self.analysis_name = "Green Kubo Self-Diffusion Coefficients"

        self.result_keys = ["diffusion_coefficient", "uncertainty"]
        self.result_series_keys = ["time", "acf"]

    @call
    def __call__(
        self,
        plot: bool = True,
        species: list = None,
        data_range: int = 500,
        save: bool = True,
        correlation_time: int = 1,
        atom_selection=np.s_[:],
        molecules: bool = False,
        gpu: bool = False,
        tau_values: Union[int, List, Any] = np.s_[:],
        integration_range: int = None,
    ):
        """
        Constructor for the Green-Kubo diffusion coefficients class.

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
        molecules : bool
                If true, molecules are used instead of atoms.
        gpu : bool
                If true, scale the memory requirement down to the amount of
                the biggest GPU in the system.
        integration_range : int
                Range over which to integrate. Default is to integrate over
                the full data range.
        """
        if species is None:
            if molecules:
                species = list(self.experiment.molecules)
            else:
                species = list(self.experiment.species)
        if integration_range is None:
            integration_range = data_range

        # set args that will affect the computation result
        self.args = Args(
            data_range=data_range,
            correlation_time=correlation_time,
            atom_selection=atom_selection,
            tau_values=tau_values,
            molecules=molecules,
            species=species,
            integration_range=integration_range
        )

        self.gpu = gpu
        self.plot = plot
        self.time = self._handle_tau_values()
        self.vacf = np.zeros(self.data_resolution)
        self.sigma = []

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
        if self.args.molecules:
            numerator = self.experiment.units["length"] ** 2
            denominator = (
                3
                * self.experiment.units["time"]
                * (self.args.integration_range - 1)
                * len(self.experiment.molecules[species]["indices"])
            )
            self.prefactor = numerator / denominator
        else:
            numerator = self.experiment.units["length"] ** 2
            denominator = (
                3
                * self.experiment.units["time"]
                * self.args.integration_range
                * len(self.experiment.species[species]["indices"])
            )
            self.prefactor = numerator / denominator

    def ensemble_operation(self, ensemble):
        """
        Calculate and return the msd.

        Parameters
        ----------
        ensemble

        Returns
        -------
        MSD of the tensor_values.
        """
        vacf = self.args.data_range * tfp.stats.auto_correlation(
            ensemble, normalize=False, axis=1, center=False
        )

        vacf = tf.reduce_sum(tf.reduce_sum(vacf, axis=0), -1)
        self.vacf += vacf
        self.sigma.append(
            np.trapz(
                vacf[: self.args.integration_range], x=self.time[: self.args.integration_range]
            )
        )

    def plot_data(self, data):
        """Plot the data"""
        for selected_species, val in data.items():
            span = Span(
                location=(
                    np.array(val[self.result_series_keys[0]])
                    * self.experiment.units["time"]
                )[self.args.integration_range - 1],
                dimension="height",
                line_dash="dashed",
            )
            self.run_visualization(
                x_data=np.array(val[self.result_series_keys[0]])
                * self.experiment.units["time"],
                y_data=np.array(val[self.result_series_keys[1]]),
                title=(
                    f"{val[self.result_keys[0]]: 0.3E} +-"
                    f" {val[self.result_keys[1]]: 0.3E}"
                ),
                layouts=[span],
            )

    def postprocessing(self, species: str):
        """
        Apply post-processing to the data.

        Parameters
        ----------
        species : str
                Current species on which you are operating.

        Returns
        -------

        """
        result = self.prefactor * np.array(self.sigma)

        data = {
            self.result_keys[0]: np.mean(result).tolist(),
            self.result_keys[1]: (np.std(result) / np.sqrt(len(result))).tolist(),
            self.result_series_keys[0]: self.time.tolist(),
            self.result_series_keys[1]: self.vacf.numpy().tolist(),
        }

        self.queue_data(data=data, subjects=[species])

    def run_calculator(self):
        """
        Run analysis.

        Returns
        -------

        """
        # Loop over species
        for species in self.args.species:
            dict_ref = str.encode("/".join([species, self.loaded_property[0]]))
            self._calculate_prefactor(species)

            batch_ds = self.get_batch_dataset([species])

            for batch in tqdm(
                batch_ds,
                ncols=70,
                desc=species,
                total=self.n_batches,
                disable=self.memory_manager.minibatch,
            ):
                ensemble_ds = self.get_ensemble_dataset(batch, species, split=True)

                for ensemble in ensemble_ds:
                    self.ensemble_operation(ensemble[dict_ref])

            # Scale, save, and plot the data.
            self.postprocessing(species)
