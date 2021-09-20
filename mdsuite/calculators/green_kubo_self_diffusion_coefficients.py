"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html.

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Summary
-------
This module contains the code for the Green-Kubo diffusion coefficient class.
This class is called by the Experiment class and instantiated when the user
calls the Experiment.einstein_diffusion_coefficients method. The methods in
class can then be called by the Experiment.green_kubo_diffusion_coefficients
method and all necessary calculations performed.
"""
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
from bokeh.models import Span
from mdsuite.calculators.calculator import Calculator, call
from mdsuite.database.calculator_database import Parameters


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
    experiment.run_computation.GreenKuboSelfDiffusionCoefficients(data_range=500, plot=True, correlation_time=10)
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

        self.loaded_property = "Velocities"
        self.scale_function = {"linear": {"scale_factor": 150}}

        self.database_group = "Diffusion_Coefficients"
        self.x_label = "$$Time (s)$$"
        self.y_label = "$$VACF (m^{2}/s^{2})$$"
        self.analysis_name = "Green Kubo Self-Diffusion Coefficients"

    @call
    def __call__(
        self,
        plot: bool = False,
        species: list = None,
        data_range: int = 500,
        save: bool = True,
        correlation_time: int = 1,
        atom_selection=np.s_[:],
        export: bool = False,
        molecules: bool = False,
        gpu: bool = False,
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
        export : bool
                If true, export the data directly into a csv file.
        molecules : bool
                If true, molecules are used instead of atoms.
        gpu : bool
                If true, scale the memory requirement down to the amount of
                the biggest GPU in the system.
        integration_range : int
                Range over which to integrate. Default is to integrate over
                the full data range.
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

        self.molecules = molecules
        self.species = species  # Which species to calculate for

        self.vacf = np.zeros(self.data_range)
        self.sigma = []

        if integration_range is None:
            self.integration_range = self.data_range
        else:
            self.integration_range = integration_range

        if species is None:
            if molecules:
                self.species = list(self.experiment.molecules)
            else:
                self.species = list(self.experiment.species)

    def _update_output_signatures(self):
        """
        After having run _prepare managers, update the output signatures.

        Returns
        -------
        Update the class state.
        """
        # Update the batch update signature for database loading.
        self.batch_output_signature = tf.TensorSpec(
            shape=(None, self.batch_size, 3), dtype=tf.float64
        )

        # Update ensemble output signature for ensemble loading.
        self.ensemble_output_signature = tf.TensorSpec(
            shape=(None, self.data_range, 3), dtype=tf.float64
        )

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
        if self.molecules:
            numerator = self.experiment.units["length"] ** 2
            denominator = (
                3
                * self.experiment.units["time"]
                * (self.integration_range - 1)
                * len(self.experiment.molecules[species]["indices"])
            )
            self.prefactor = numerator / denominator
        else:
            numerator = self.experiment.units["length"] ** 2
            denominator = (
                3
                * self.experiment.units["time"]
                * self.integration_range
                * len(self.experiment.species[species]["indices"])
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
        vacf = self.data_range * tfp.stats.auto_correlation(
            ensemble, normalize=False, axis=1, center=False
        )

        vacf = tf.reduce_sum(tf.reduce_sum(vacf, axis=0), -1)
        self.vacf += vacf
        self.sigma.append(
            np.trapz(
                vacf[: self.integration_range], x=self.time[: self.integration_range]
            )
        )

    def _post_operation_processes(self, species: str = None):
        """
        Apply post-op processes such as saving and plotting.

        Returns
        -------

        """
        result = self.prefactor * np.array(self.sigma)

        properties = Parameters(
            Property=self.database_group,
            Analysis=self.analysis_name,
            data_range=self.data_range,
            data=[{'diffusion_coefficient': result[0],
                   'uncertainty': result[1]}],
            Subject=[species]
        )
        data = properties.data
        data += [{'time': x, 'acf': y} for x, y in
                 zip(self.time, self.vacf)]
        properties.data = data
        self.update_database(properties)

        # Update the plot if required
        if self.plot:
            span = Span(
                location=(np.array(self.time) * self.experiment.units["time"])[self.integration_range - 1],
                dimension='height',
                line_dash='dashed'
            )
            self.run_visualization(
                x_data=np.array(self.time) * self.experiment.units['time'],
                y_data=self.vacf.numpy(),
                title=f"{species}: {np.mean(result): .3E} +- {np.std(result) / (np.sqrt(len(result))): .3E}",
                layouts=[span]
            )
