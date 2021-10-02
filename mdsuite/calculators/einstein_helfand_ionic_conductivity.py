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
MDSuite module for the computation of ionic conductivity using the Einstein method.
"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from mdsuite.calculators.calculator import Calculator, call
from mdsuite.utils.units import elementary_charge, boltzmann_constant
from dataclasses import dataclass
from mdsuite.database import simulation_properties


@dataclass
class Args:
    data_range: int
    correlation_time: int
    tau_values: np.s_
    atom_selection: np.s_


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

        self.loaded_property = simulation_properties.translational_dipole_moment
        self.dependency = simulation_properties.unwrapped_positions
        self.system_property = True

        self.database_group = "Ionic_Conductivity"
        self.x_label = r"$$\text{Time} / s$$"
        self.y_label = r"$$\text{MSD} / m^2/s$$"
        self.analysis_name = "Einstein Helfand Ionic Conductivity"
        self.prefactor: float
        self.trial_pp = True

        self.result_keys = ["ionic_conductivity", "uncertainty"]
        self.result_series_keys = ["time", "msd"]

    @call
    def __call__(
        self,
        plot=True,
        data_range=500,
        save=True,
        correlation_time=1,
        tau_values: np.s_ = np.s_[:],
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
        gpu : bool
                If true, reduce memory usage to the maximum GPU capability.

        """

        # set args that will affect the computation result
        self.args = Args(
            data_range=data_range,
            correlation_time=correlation_time,
            tau_values=tau_values,
            atom_selection=np.s_[:],
        )

        self.gpu = gpu
        self.time = self._handle_tau_values()
        self.msd_array = np.zeros(self.data_resolution)

    def check_input(self):
        pass

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
        msd = tf.math.squared_difference(
            tf.gather(ensemble, self.args.tau_values, axis=0), ensemble[None, 0]
        )
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

    def run_calculator(self):
        """
        Run analysis.

        Returns
        -------

        """
        # Compute the pre-factor early.
        self._calculate_prefactor()

        dict_ref = str.encode(
            "/".join([self.loaded_property[0], self.loaded_property[0]])
        )

        batch_ds = self.get_batch_dataset([self.loaded_property[0]])

        for batch in tqdm(
            batch_ds,
            ncols=70,
            total=self.n_batches,
            disable=self.memory_manager.minibatch,
        ):
            ensemble_ds = self.get_ensemble_dataset(batch, self.loaded_property[0])

            for ensemble in ensemble_ds:
                self.ensemble_operation(ensemble[dict_ref])

        # Scale, save, and plot the data.
        self._apply_averaging_factor()
        self._post_operation_processes()
