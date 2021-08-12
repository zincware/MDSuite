"""
This program and the accompanying materials are made available under the
terms of the Eclipse Public License v2.0 which accompanies this distribution
and is available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Class for the calculation of the Green-Kubo viscosity.

Summary
This module contains the code for the Green-Kubo viscosity class.
This class is called by the experiment class.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from mdsuite.calculators.calculator import Calculator


class GreenKuboViscosity(Calculator):
    """Class for the Green-Kubo ionic conductivity implementation

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
    experiment.run_computation.GreenKuboViscosity(data_range=500, plot=True,
    correlation_time=10)
    """

    def __init__(self, experiment):
        """

        Attributes
        ----------
        experiment :  object
                Experiment class to call from
        """
        super().__init__(experiment)
        self.scale_function = {"linear": {"scale_factor": 5}}

        self.loaded_property = "Momentum_Flux"
        self.database_group = "Viscosity"
        self.system_property = True

        self.x_label = "Time (s)"
        self.y_label = r"SACF ($C^{2}\cdot m^{2}/s^{2}$)"
        self.analysis_name = "Green_Kubo_Viscosity"
        self.prefactor: float

    def __call__(
        self,
        plot=False,
        data_range=500,
        save=True,
        correlation_time: int = 1,
        export: bool = False,
        gpu: bool = False,
        integration_range: int = None,
    ):
        """

        Attributes
        ----------
        plot : bool
                if true, plot the tensor_values
        data_range :
                Number of configurations to use in each ensemble
        save :
                If true, tensor_values will be saved after the analysis
        """
        self.update_user_args(
            plot=plot,
            data_range=data_range,
            save=save,
            correlation_time=correlation_time,
            export=export,
            gpu=gpu,
        )

        self.jacf = np.zeros(self.data_range)
        self.sigma = []

        if integration_range is None:
            self.integration_range = self.data_range
        else:
            self.integration_range = integration_range

        out = self.run_analysis()
        self.experiment.save_class()

        return out

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
        Compute the ionic conductivity prefactor.

        Parameters
        ----------
        species

        Returns
        -------

        """
        # prepare the prefactor for the integral
        numerator = 1  # self.experiment.volume
        denominator = (
            3
            * (self.data_range - 1)
            * self.experiment.temperature
            * self.experiment.units["boltzman"]
            * self.experiment.volume
        )

        prefactor_units = (
            self.experiment.units["pressure"] ** 2
            * self.experiment.units["length"] ** 3
            * self.experiment.units["time"]
            / self.experiment.units["energy"]
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
        ensemble : tf.Tensor
                An ensemble of data to be studied.
        index : int

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

        properties = {
            "Property": self.database_group,
            "Analysis": self.analysis_name,
            "Subject": ["System"],
            "data_range": self.data_range,
            "data": [
                {
                    "x": np.mean(result),
                    "uncertainty": np.std(result) / (np.sqrt(len(result))),
                }
            ],
        }
        self._update_properties_file(properties)
        # Update the plot if required
        if self.plot:
            plt.plot(np.array(self.time) * self.experiment.units["time"], self.jacf)
            plt.vlines(
                (np.array(self.time) * self.experiment.units["time"])[
                    self.integration_range
                ],
                min(self.jacf),
                max(self.jacf),
            )
            self._plot_data()

        if self.save:
            properties = {
                "Property": self.database_group,
                "Analysis": self.analysis_name,
                "Subject": ["System"],
                "data_range": self.data_range,
                "data": [{"x": x, "y": y} for x, y in zip(self.time, self.jacf)],
                "information": "series",
            }
            self._update_properties_file(properties)
        if self.export:
            self._export_data(
                name=self._build_table_name("System"),
                data=self._build_pandas_dataframe(self.time, self.jacf),
            )
