"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Class for the calculation of the Green-Kubo thermal conductivity.

Summary
-------
This module contains the code for the Green-Kubo thermal conductivity class.
This class is called by the Experiment class and instantiated when the user
calls the Experiment.green_kubo_thermal_conductivity method. The methods in
class can then be called by the Experiment.green_kubo_thermal_conductivity
 method and all necessary calculations performed.
"""
import warnings
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from mdsuite.calculators.calculator import Calculator, call
import tensorflow_probability as tfp
from bokeh.models import Span

tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboThermalConductivity(Calculator):
    """
    Class for the Green-Kubo Thermal conductivity implementation

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

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.GreenKuboThermalConductivity(data_range=500,
    plot=True, correlation_time=10)
    """

    def __init__(self, **kwargs):
        """
        Class for the Green-Kubo Thermal conductivity implementation

        Attributes
        ----------
        experiment :  object
                Experiment class to call from
        """
        super().__init__(**kwargs)
        self.scale_function = {"linear": {"scale_factor": 5}}

        self.loaded_property = "Thermal_Flux"
        self.database_group = "Thermal_Conductivity"
        self.system_property = True

        self.x_label = r"$$\text{Time} / s$$"
        self.y_label = r"$$\text{JACF} / ($C^{2}\cdot m^{2}/s^{2}$$"
        self.analysis_name = "Green_Kubo_Thermal_Conductivity"

    @call
    def __call__(
        self,
        plot=False,
        data_range=500,
        save=True,
        correlation_time: int = 1,
        gpu: bool = False,
        integration_range: int = None,
    ):
        """
        Class for the Green-Kubo Thermal conductivity implementation

        Parameters
        ----------
        plot : bool
                if true, plot the output.
        data_range : int
                Data range to use in the analysis.
        save : bool
                if true, save the output.
        correlation_time : int
                Correlation time to use in the window sampling.
        export : bool
                If true, export the data directly into a csv file.
        gpu : bool
                If true, scale the memory requirement down to the amount of
                the biggest GPU in the system.
        integration_range : int
                Range over which the integration should be performed.
        """
        self.update_user_args(
            plot=plot,
            data_range=data_range,
            save=save,
            correlation_time=correlation_time,
            gpu=gpu,
        )

        self.jacf = np.zeros(self.data_range)
        self.prefactor: float
        self.sigma = []

        if integration_range is None:
            self.integration_range = self.data_range
        else:
            self.integration_range = integration_range

    def _update_output_signatures(self):
        """
        Update the output signature for the IC.

        Returns
        -------

        """
        self.batch_output_signature = tf.TensorSpec(
            shape=(self.batch_size, 3), dtype=tf.float64
        )
        self.ensemble_output_signature = tf.TensorSpec(
            shape=(self.data_range, 3), dtype=tf.float64
        )

    def _calculate_prefactor(self, species: str = None):
        """
        Compute the ionic conductivity pre-factor.

        Parameters
        ----------
        species

        Returns
        -------

        """
        # Calculate the prefactor
        # prepare the prefactor for the integral
        numerator = 1
        denominator = (
            3
            * (self.data_range - 1)
            * self.experiment.temperature ** 2
            * self.experiment.units["boltzman"]
                      * self.experiment.volume
        )
        prefactor_units = (
            self.experiment.units["energy"]
            / self.experiment.units["length"]
            / self.experiment.units["time"]
        )

        self.prefactor = (numerator / denominator) * prefactor_units

    def _apply_averaging_factor(self):
        """
        Apply the averaging factor to the msd array.
        Returns
        -------

        """
        pass

    def _apply_operation(self, ensemble: tf.Tensor, index: int):
        """
        Calculate and return the msd.

        Parameters
        ----------
        ensemble : tf.Tenor
                Ensemble to analyze.

        Returns
        -------
        MSD of the tensor_values.
        """
        jacf = self.data_range * tf.reduce_sum(
            tfp.stats.auto_correlation(ensemble, normalize=False, axis=0, center=False),
            axis=-1,
        )
        self.jacf += jacf
        self.sigma.append(
            np.trapz(
                jacf[: self.integration_range], x=self.time[: self.integration_range]
            )
        )

    def _post_operation_processes(self, species: str = None):
        """
        call the post-op processes

        Returns
        -------

        """
        result = self.prefactor * np.array(self.sigma)

        data = {
            "computation_results": result[0],
            "uncertainty": result[1],
            'time': self.time.tolist(),
            'acf': self.jacf.numpy().tolist()
        }

        self.queue_data(data=data, subjects=['System'])

        # Update the plot if required
        if self.plot:
            span = Span(
                location=(np.array(self.time) * self.experiment.units["time"])[
                    self.integration_range - 1],
                dimension='height',
                line_dash='dashed'
            )
            self.run_visualization(
                x_data=np.array(self.time) * self.experiment.units['time'],
                y_data=self.jacf.numpy(),
                title=f"{result[0]} +- {result[1]}",
                layouts=[span]
            )
