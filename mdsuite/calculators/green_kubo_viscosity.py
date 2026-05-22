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
MDSuite module for the computation of the viscosity in a system using the Green-Kubo
relation as applied to the momentum flux measured during a simulation.
"""
from abc import ABC
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from bokeh.models import Span
from tqdm import tqdm

from mdsuite.calculators.trajectory_calculator import TrajectoryCalculator
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.utils import DatasetKeys
from mdsuite.utils.calculator_helper_methods import auto_correlation


@dataclass
class Args:
    """Data class for the saved properties."""

    data_range: int
    correlation_time: int
    tau_values: np.s_
    atom_selection: np.s_
    integration_range: int


class GreenKuboViscosity(TrajectoryCalculator, ABC):
    """Class for the Green-Kubo ionic conductivity implementation.

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

    def __init__(
        self,
        plot=False,
        data_range=500,
        tau_values: np.s_ = np.s_[:],
        correlation_time: int = 1,
        integration_range: int = None,
    ):
        """Green-Kubo viscosity calculator (momentum-flux form).

        Parameters
        ----------
        plot : bool
                if true, plot the tensor_values.
        data_range :
                Number of configurations to use in each ensemble.
        tau_values : np.s_
                Selection of tau values to use in the window sliding.
        correlation_time : int
                Correlation time to use in the window sampling.
        integration_range : int
                Range over which integration should be performed; ``None`` means
                ``data_range``.
        """
        super().__init__()
        self.scale_function = {"linear": {"scale_factor": 5}}

        self.loaded_property = mdsuite_properties.momentum_flux
        self.system_property = True

        self.x_label = r"$$\text{Time} / s$$"
        self.y_label = r"\text{SACF} / C^{2}\cdot m^{2}/s^{2}$$"
        self.analysis_name = "Green_Kubo_Viscosity"
        self.prefactor = None
        self._dtype = tf.float64

        self.plot = plot

        # Args is locked in at construction — no experiment access needed.
        if integration_range is None:
            integration_range = data_range
        self.args = Args(
            data_range=data_range,
            correlation_time=correlation_time,
            tau_values=tau_values,
            atom_selection=np.s_[:],
            integration_range=integration_range,
        )

    def _setup(self):
        """Experiment-dependent per-run state."""
        self.time = self._handle_tau_values()
        self.jacf = np.zeros(self.data_resolution)
        self.sigma = []

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
        # prepare the prefactor for the integral
        numerator = 1  # self.experiment.volume
        denominator = (
            3
            * (self.resolved_data_range - 1)
            * self.experiment.temperature
            * self.experiment.units.boltzmann
            * self.experiment.volume
        )

        prefactor_units = (
            self.experiment.units.pressure**2
            * self.experiment.units.volume
            * self.experiment.units.time
            / self.experiment.units.energy
        )

        self.prefactor = (numerator / denominator) * prefactor_units

    def _apply_averaging_factor(self):
        """
        Apply the averaging factor to the msd array.

        Returns
        -------
        -------.

        """
        pass

    def ensemble_operation(self, ensemble: tf.Tensor):
        """Accumulate the momentum-flux autocorrelation for one window.

        Uses the JAX-vmap :func:`auto_correlation` helper to compute the
        unbiased ACF summed over particles and Cartesian components.
        """
        jacf = auto_correlation(np.asarray(ensemble))
        self.jacf += jacf
        self.sigma.append(
            np.trapezoid(
                jacf[: self.args.integration_range],
                x=self.time[: self.args.integration_range],
            )
        )

    def _post_operation_processes(self):
        """Aggregate per-ensemble integrals into the final viscosity.

        Mean across windows for the value, standard error of the mean for the
        uncertainty.
        """
        result = self.prefactor * np.array(self.sigma)
        viscosity = float(np.mean(result))
        viscosity_SEM = (
            float(np.std(result) / np.sqrt(len(result))) if len(result) > 1 else 0.0
        )

        acf_array = np.asarray(self.jacf)
        data = {
            "viscosity": [viscosity],
            "uncertainty": [viscosity_SEM],
            "time": self.time.tolist(),
            "acf": acf_array.tolist(),
        }

        self.queue_data(data=data, subjects=["System"])

        if self.plot:
            span = Span(
                location=(np.array(self.time) * self.experiment.units.time)[
                    self.args.integration_range - 1
                ],
                dimension="height",
                line_dash="dashed",
            )
            self.run_visualization(
                x_data=np.array(self.time) * self.experiment.units.time,
                y_data=acf_array,
                title=f"{viscosity} +- {viscosity_SEM}",
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
            "/".join([DatasetKeys.OBSERVABLES, self.loaded_property.name])
        )

        batch_ds = self.get_batch_dataset([DatasetKeys.OBSERVABLES])

        for batch in tqdm(
            batch_ds,
            ncols=70,
            total=self.n_batches,
            disable=self.memory_manager.minibatch,
        ):
            ensemble_ds = self.get_ensemble_dataset(batch, DatasetKeys.OBSERVABLES)

            for ensemble in ensemble_ds:
                self.ensemble_operation(ensemble[dict_ref])

        # Scale, save, and plot the data.
        self._apply_averaging_factor()
        self._post_operation_processes()
