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
import warnings
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from mdsuite.calculators.calculator import Calculator, call
from mdsuite.utils.units import elementary_charge, boltzmann_constant

tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class EinsteinHelfandIonicConductivity(Calculator):
    """
    Class for the Einstein-Helfand Ionic Conductivity

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
    experiment.run_computation.EinsteinHelfandTIonicConductivity(data_range=500,
                                                                 plot=True,
                                                                 correlation_time=10)
    """

    def __init__(self, **kwargs):
        """
        Python constructor

        Parameters
        ----------
        experiment :  object
            Experiment class to call from
        """

        # parse to the experiment class
        super().__init__(**kwargs)
        self.scale_function = {"linear": {"scale_factor": 5}}

        self.loaded_property = (  # Property to be loaded for the analysis
            "Translational_Dipole_Moment"
        )
        self.dependency = "Unwrapped_Positions"
        self.system_property = True

        self.database_group = (
            "Ionic_Conductivity"
        )
        self.x_label = r"$$\text{Time} / s$$"
        self.y_label = r"$$\text{MSD} / m^2/s$$"
        self.analysis_name = "Einstein Helfand Ionic Conductivity"
        self.prefactor: float

        self.result_keys = ["ionic_conductivity", "uncertainty"]
        self.result_series_keys = ["time", "msd"]

    @call
    def __call__(
        self,
        plot=True,
        data_range=500,
        save=True,
        correlation_time=1,
        export: bool = False,
        gpu: bool = False,
    ):
        """
        Python constructor

        Parameters
        ----------
        plot : bool
                if true, plot the tensor_values
        data_range :
                Number of configurations to use in each ensemble
        save :
                If true, tensor_values will be saved after the analysis
        correlation_time : int
                Correlation time to use in the analysis.
        export : bool
                If true, export the results to a csv.
        gpu : bool
                If true, reduce memory usage to the maximum GPU capability.

        """
        # parse to the experiment class
        self.update_user_args(
            plot=plot,
            data_range=data_range,
            save=save,
            correlation_time=correlation_time,
            export=export,
            gpu=gpu,
        )
        self.msd_array = np.zeros(self.data_range)

        return self.update_db_entry_with_kwargs(
            data_range=data_range, correlation_time=correlation_time
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
        numerator = (self.experiment.units["length"] ** 2) * (elementary_charge ** 2)
        denominator = (
            6
            * self.experiment.units["time"]
            * (self.experiment.volume * self.experiment.units["length"] ** 3)
            * self.experiment.temperature
            * boltzmann_constant
        )
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
        msd = tf.math.squared_difference(ensemble, ensemble[None, 0])
        msd = self.prefactor * tf.reduce_sum(msd, axis=1)
        self.msd_array += np.array(msd)  # Update the averaged function

    def _post_operation_processes(self, species: str = None):
        """
        call the post-op processes
        Returns
        -------

        """
        result = self._fit_einstein_curve([self.time, self.msd_array])

        data = {
            self.result_keys[0]: result[0].tolist(),
            self.result_keys[1]: result[1].tolist(),
            self.result_series_keys[0]: self.time.tolist(),
            self.result_series_keys[1]: self.msd_array.tolist(),
        }

        self.queue_data(data=data, subjects=["System"])
