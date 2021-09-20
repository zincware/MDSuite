"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Class for the calculation of viscosity.

Summary
-------
This module contains the code for the viscosity class. This class is called by the
Experiment class and instantiated when the user calls the ... method.
The methods in class can then be called by the ... method and all necessary
calculations performed.
"""
import warnings
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp
from mdsuite.database.calculator_database import Parameters
from mdsuite.calculators.calculator import Calculator, call

tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


# Set style preferences, turn off warning, and suppress the duplication of loading bars.
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboViscosityFlux(Calculator):
    """
    Class for the Green Kubo viscosity from flux implementation

    Attributes
    ----------
    experiment :  object
            Experiment class to call from

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.GreenKuboViscosityFlux(data_range=500, plot=True, correlation_time=10)
    """

    def __init__(self, **kwargs):
        """
        Python constructor for the experiment class.

        Parameters
        ----------
        experiment : object
                Experiment class to read and write to
        """
        super().__init__(**kwargs)
        self.scale_function = {'linear': {'scale_factor': 5}}

        self.loaded_property = 'Stress_visc'  # Property to be loaded for the analysis
        self.system_property = True

        self.database_group = 'Viscosity'  # Which database_path group to save the tensor_values in
        self.analysis_name = 'Viscosity_Flux'
        self.x_label = 'Time (s)'
        self.y_label = 'JACF ($C^{2}\\cdot m^{2}/s^{2}$)'

        self.prefactor: float
        self.jacf = np.zeros(self.data_range)
        self.sigma = []

    @call
    def __call__(self,
                 plot=False,
                 data_range=500,
                 correlation_time=1,
                 save=True,
                 gpu: bool = False,
                 integration_range: int = None):
        """
        Python constructor for the experiment class.

        Parameters
        ----------
        plot : bool
                If true, a plot of the analysis is saved.
        data_range : int
                Number of configurations to include in each ensemble
        """

        self.update_user_args(
            plot=plot, data_range=data_range, save=save,
            correlation_time=correlation_time, gpu=gpu)

        self.jacf = np.zeros(self.data_range)
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
        self.batch_output_signature = tf.TensorSpec(shape=(self.batch_size, 3), dtype=tf.float64)
        self.ensemble_output_signature = tf.TensorSpec(shape=(self.data_range, 3), dtype=tf.float64)

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
        numerator = self.experiment.volume
        denominator = 3 * (self.data_range - 1) * self.experiment.temperature * self.experiment.units['boltzman']

        prefactor_units = self.experiment.units['pressure'] ** 2 * self.experiment.units['length'] ** 3 * \
            self.experiment.units['time'] / self.experiment.units['energy']

        self.prefactor = (numerator / denominator)*prefactor_units

    def _apply_averaging_factor(self):
        """
        Apply the averaging factor to the msd array.
        Returns
        -------

        """
        self.jacf /= max(self.jacf)

    def _apply_operation(self, ensemble, index):
        """
        Calculate and return the vacf.

        Parameters
        ----------
        ensemble

        Returns
        -------
        updates class vacf with the tensor_values.
        """
        jacf = self.data_range * tf.reduce_sum(
            tfp.stats.auto_correlation(ensemble, normalize=False, axis=0,
                                       center=False),
            axis=-1,
        )
        self.jacf += jacf[int(self.data_range - 1):]
        self.sigma.append(
            np.trapz(
                jacf[: self.integration_range],
                x=self.time[: self.integration_range]
            )
        )

    def _post_operation_processes(self, species: str = None):
        """
        call the post-op processes
        Returns
        -------

        """
        result = self.prefactor*np.array(self.sigma)

        properties = Parameters(
            Property=self.database_group,
            Analysis=self.analysis_name,
            data_range=self.data_range,
            data=[{'viscosity': result[0],
                   'uncertainty': result[1]}],
            Subject=["System"]
        )
        data = properties.data
        data += [{'time': x, 'acf': y} for x, y in
                 zip(self.time, self.jacf)]
        properties.data = data
        self.update_database(properties)

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
