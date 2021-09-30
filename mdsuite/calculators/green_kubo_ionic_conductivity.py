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
import tensorflow as tf
import tensorflow_probability as tfp
from mdsuite.utils.units import boltzmann_constant, elementary_charge
from mdsuite.calculators.calculator import Calculator, call
from bokeh.models import Span
from mdsuite.database.scheme import Computation


class GreenKuboIonicConductivity(Calculator):
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

        self.loaded_property = "Ionic_Current"
        self.system_property = True

        self.database_group = "Ionic_Conductivity"
        self.x_label = r"$$\text{Time} / s"
        self.y_label = r"$$\text{JACF} / C^{2}\cdot m^{2}/s^{2}$$"
        self.analysis_name = "Green_Kubo_Ionic_Conductivity"

        self.result_keys = ['ionic_conductivity', 'uncertainty']
        self.result_series_keys = ['time', 'acf']

        self.prefactor: float

    @call
    def __call__(
            self,
            plot=True,
            data_range=500,
            save=True,
            correlation_time=1,
            export: bool = False,
            gpu: bool = False,
            integration_range: int = None,
    ) -> Computation:
        """

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
                Range over which integration should be performed.
        """

        # update experiment class
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

        return self.update_db_entry_with_kwargs(
            data_range=data_range,
            correlation_time=correlation_time
        )

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
        # Calculate the prefactor
        numerator = (elementary_charge ** 2) * (self.experiment.units["length"] ** 2)
        denominator = (
                3
                * boltzmann_constant
                * self.experiment.temperature
                * self.experiment.volume
                * (self.experiment.units["length"] ** 3)
                * self.data_range
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
            self.result_keys[0]: np.mean(result).tolist(),
            self.result_keys[1]: (np.std(result) / np.sqrt(len(result))).tolist(),
            self.result_series_keys[0]: self.time.tolist(),
            self.result_series_keys[1]: self.jacf.numpy().tolist()
        }

        self.queue_data(data=data, subjects=['System'])

    def plot_data(self, data):
        """Plot the data"""
        for selected_species, val in data.items():
            span = Span(
                location=(np.array(val[self.result_series_keys[0]]) * self.experiment.units["time"])[
                    self.integration_range - 1],
                dimension='height',
                line_dash='dashed'
            )
            self.run_visualization(
                x_data=np.array(val[self.result_series_keys[0]]) * self.experiment.units['time'],
                y_data=np.array(val[self.result_series_keys[1]]),
                title=f"{val[self.result_keys[0]]: 0.3E} +- {val[self.result_keys[1]]: 0.3E}",
                layouts=[span]
            )
