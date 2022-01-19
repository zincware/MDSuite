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
MDSuite module for the computation of the ionic conductivity of a system using the
Green-Kubo relation. Ionic conductivity describes how well a system can conduct an
electrical charge due to the mobility of the ions contained within it. This differs
from electronic conductivity which is transferred by electrons.
"""
from abc import ABC
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from bokeh.models import Span
from tqdm import tqdm

from mdsuite.calculators.calculator import call
from mdsuite.calculators.trajectory_calculator import TrajectoryCalculator
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.utils.units import boltzmann_constant, elementary_charge


@dataclass
class Args:
    """
    Data class for the saved properties.
    """

    data_range: int
    correlation_time: int
    tau_values: np.s_
    atom_selection: np.s_
    integration_range: int


class GreenKuboIonicConductivity(TrajectoryCalculator, ABC):
    """
    Class for the Green-Kubo ionic conductivity implementation

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
    experiment.run_computation.GreenKuboIonicConductivity(data_range=500,
    plot=True, correlation_time=10)
    """

    def __init__(self, **kwargs):
        """

        Attributes
        ----------
        experiment :  object
                Experiment class to call from
        """

        # update experiment class
        super().__init__(**kwargs)
        self.scale_function = {"linear": {"scale_factor": 5}}

        self.loaded_property = mdsuite_properties.ionic_current
        self.system_property = True

        self.x_label = r"$$\text{Time} / s$$"
        self.y_label = r"$$\text{JACF} / C^{2}\cdot m^{2}/s^{2}$$"
        self.analysis_name = "Green_Kubo_Ionic_Conductivity"

        self.result_keys = ["ionic_conductivity", "uncertainty"]
        self.result_series_keys = ["time", "acf"]

        self.prefactor = None
        self._dtype = tf.float64

    @call
    def __call__(
        self,
        plot=True,
        data_range=500,
        correlation_time=1,
        tau_values: np.s_ = np.s_[:],
        gpu: bool = False,
        integration_range: int = None,
    ):
        """

        Parameters
        ----------
        plot : bool
                if true, plot the output.
        data_range : int
                Data range to use in the analysis.
        correlation_time : int
                Correlation time to use in the window sampling.
        gpu : bool
                If true, scale the memory requirement down to the amount of
                the biggest GPU in the system.
        integration_range : int
                Range over which integration should be performed.
        """

        self.gpu = gpu
        self.plot = plot
        self.jacf: np.ndarray
        self.sigma = []

        if integration_range is None:
            integration_range = data_range

        # set args that will affect the computation result
        self.args = Args(
            data_range=data_range,
            correlation_time=correlation_time,
            tau_values=tau_values,
            atom_selection=np.s_[:],
            integration_range=integration_range,
        )

        self.time = self._handle_tau_values()
        self.jacf = np.zeros(self.data_resolution)

    def check_input(self):
        """
        Check the user input to ensure no conflicts are present.

        Returns
        -------

        """
        self._run_dependency_check()

    def _calculate_prefactor(self):
        """
        Compute the ionic conductivity prefactor.

        Returns
        -------

        """
        # Calculate the prefactor
        numerator = (elementary_charge ** 2) * (self.experiment.units["length"] ** 2)
        denominator = (
            3
            * boltzmann_constant
            * self.experiment.temperature
            * self.experiment.volume
            * (self.experiment.units["length"] ** 3)
            * self.args.data_range
            * self.experiment.units["time"]
        )
        self.prefactor = numerator / denominator

    def _apply_averaging_factor(self):
        """
        Apply the averaging factor to the msd array.
        Returns
        -------

        """
        pass

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
        jacf = self.args.data_range * tf.reduce_sum(
            tfp.stats.auto_correlation(
                tf.gather(ensemble, self.args.tau_values, axis=1),
                normalize=False,
                axis=1,
                center=False,
            ),
            axis=-1,
        )[0, :]
        self.jacf += jacf
        self.sigma.append(
            np.trapz(
                jacf,
                x=self.time[self.args.tau_values],
            )
        )

    def _post_operation_processes(self):
        """
        call the post-op processes
        Returns
        -------

        """
        result = self.prefactor * np.array(self.sigma)

        data = {
            self.result_keys[0]: np.mean(result).tolist(),
            self.result_keys[1]: (np.std(result) / np.sqrt(len(result))).tolist(),
            self.result_series_keys[0]: self.time.tolist(),
            self.result_series_keys[1]: self.jacf.numpy().tolist(),
        }

        self.queue_data(data=data, subjects=["System"])

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

    def run_calculator(self):
        """
        Run analysis.

        Returns
        -------

        """
        self.check_input()
        # Compute the pre-factor early.
        self._calculate_prefactor()

        dict_ref = str.encode(
            "/".join([self.loaded_property.name, self.loaded_property.name])
        )

        batch_ds = self.get_batch_dataset([self.loaded_property.name])
        for batch in tqdm(
            batch_ds,
            ncols=70,
            total=self.n_batches,
            disable=self.memory_manager.minibatch,
        ):
            ensemble_ds = self.get_ensemble_dataset(batch, self.loaded_property.name)
            for ensemble in ensemble_ds:
                self.ensemble_operation(ensemble[dict_ref])

        # Scale, save, and plot the data.
        self._apply_averaging_factor()
        self._post_operation_processes()
