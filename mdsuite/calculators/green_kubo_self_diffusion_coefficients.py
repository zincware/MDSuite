"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Class for the calculation of the Green-Kubo diffusion coefficients.
Summary
-------
This module contains the code for the Einstein diffusion coefficient class. This class is called by the
Experiment class and instantiated when the user calls the Experiment.einstein_diffusion_coefficients method.
The methods in class can then be called by the Experiment.green_kubo_diffusion_coefficients method and all necessary
calculations performed.
"""
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import warnings
from tqdm import tqdm
import tensorflow as tf
from mdsuite.calculators.calculator import Calculator

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class GreenKuboSelfDiffusionCoefficients(Calculator):
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

        self.loaded_property = 'Velocities'  # Property to be loaded for the analysis
        self.scale_function = {'linear': {'scale_factor': 150}}

        self.database_group = 'Diffusion_Coefficients'  # Which database_path group to save the tensor_values in
        self.x_label = 'Time $(s)$'
        self.y_label = 'VACF $(m^{2}/s^{2})$'
        self.analysis_name = 'Green_Kubo_Self_Diffusion_Coefficients'

    def __call__(self, plot: bool = False, species: list = None, data_range: int = 500, save: bool = True,
                 correlation_time: int = 1, atom_selection=np.s_[:], export: bool = False, molecules: bool = False,
                 gpu: bool = False):
        """
        Constructor for the Green Kubo diffusion coefficients class.

        Attributes
        ----------
        plot : bool
                if true, plot the tensor_values
        species : list
                Which species to perform the analysis on
        data_range :
                Number of configurations to use in each ensemble
        save :
                If true, tensor_values will be saved after the analysis

        Returns
        -------
        data:
            A dictionary of shape {experiment_name: data} for multiple len(experiments) > 1 or otherwise just data

        """

        out = {}
        for experiment in self.experiments:
            self.experiment = experiment

            self.update_user_args(plot=plot, data_range=data_range, save=save, correlation_time=correlation_time,
                                  atom_selection=atom_selection, export=export, gpu=gpu)

            self.molecules = molecules
            self.species = species  # Which species to calculate for

            self.vacf = np.zeros(self.data_range)
            self.sigma = []

            if species is None:
                if molecules:
                    self.species = list(self.experiment.molecules)
                else:
                    self.species = list(self.experiment.species)

            if self.load_data:
                out[self.experiment.experiment_name] = self.experiment.export_property_data(
                    {"Analysis": self.analysis_name}
                )
            else:
                out[self.experiment.experiment_name] = self.run_analysis()

        if len(self.experiments) > 1:
            return out
        else:
            return out[self.experiment.experiment_name]

    def _update_output_signatures(self):
        """
        After having run _prepare managers, update the output signatures.

        Returns
        -------
        Update the class state.
        """
        self.batch_output_signature = tf.TensorSpec(shape=(None, self.batch_size, 3), dtype=tf.float64)
        self.ensemble_output_signature = tf.TensorSpec(shape=(None, self.data_range, 3), dtype=tf.float64)

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
        if self.molecules:
            # Calculate the prefactor
            numerator = self.experiment.units['length'] ** 2
            denominator = 3 * self.experiment.units['time'] * (self.data_range - 1) * \
                          len(self.experiment.molecules[species]['indices'])
            self.prefactor = numerator / denominator
        else:
            # Calculate the prefactor
            numerator = self.experiment.units['length'] ** 2
            denominator = 3 * self.experiment.units['time'] * (self.data_range - 1) * \
                          len(self.experiment.species[species]['indices'])
            self.prefactor = numerator / denominator

    def _apply_averaging_factor(self):
        """
        Apply the averaging factor to the msd array.
        Returns
        -------

        """
        self.vacf /= max(self.vacf)

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
        vacf = np.zeros(2 * self.data_range - 1)
        for item in ensemble:
            vacf += sum([signal.correlate(item[:, idx], item[:, idx], mode="full", method='auto') for idx in range(3)])

        self.vacf += vacf[int(self.data_range - 1):]  # Update the averaged function
        self.sigma.append(np.trapz(vacf[int(self.data_range - 1):], x=self.time))

    def _post_operation_processes(self, species: str = None):
        """
        Apply post-op processes such as saving and plotting.
        Returns
        -------

        """
        result = self.prefactor * np.array(self.sigma)

        properties = {"Property": self.database_group,
                      "Analysis": self.analysis_name,
                      "Subject": [species],
                      "data_range": self.data_range,
                      'data': [{'x': np.mean(result), 'uncertainty': np.std(result) / (np.sqrt(len(result)))}]
                      }
        self._update_properties_file(properties)

        # Update the plot if required
        if self.plot:
            plt.xlabel(rf'{self.x_label}')  # set the x label
            plt.ylabel(rf'{self.y_label}')  # set the y label
            plt.plot(np.array(self.time) * self.experiment.units['time'], self.vacf,
                     label=fr"{species}: {np.mean(result): .3E} $\pm$ {np.std(result) / (np.sqrt(len(result))): .3E}")

        if self.save:
            properties = {"Property": self.database_group,
                          "Analysis": self.analysis_name,
                          "Subject": [species],
                          "data_range": self.data_range,
                          'data': [{'x': x, 'y': y} for x, y in zip(self.time, self.vacf)],
                          'information': "series"
                          }
            self._update_properties_file(properties)

        if self.export:
            self._export_data(name=self._build_table_name(species), data=self._build_pandas_dataframe(self.time,
                                                                                                      self.vacf))
