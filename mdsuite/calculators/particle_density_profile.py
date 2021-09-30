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
Compute the particle density of an ion in the system.
"""
from __future__ import annotations
import logging
import numpy as np
import warnings
from typing import Union, Any, List
from tqdm import tqdm
from mdsuite.calculators.calculator import Calculator, call
import tensorflow as tf
import tensorflow_probability as tfp
from mdsuite.visualizer.d3_data_visualizer import DataVisualizer3D


tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)


class ParticleDensity(Calculator):
    """
    Class for the Einstein diffusion coefficient implementation

    Description
    -----------
    Module for the calculation of particle density for one ion species.

    Attributes
    ----------
    experiment :  Experiment
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
    experiment.run_computation.ParticleDensity(data_range=500,
                                               plot=True,
                                               correlation_time=10)
    """

    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        experiment :  Experiment
                Experiment class to call from
        experiments :  Experiment
                Experiment classes to call from
        load_data :  bool
                whether to load data or not
        """

        super().__init__(**kwargs)
        self.scale_function = {"linear": {"scale_factor": 150}}
        self.loaded_property = "Positions"
        self.species = None
        self.molecules = None
        self.database_group = "Particle_Density"
        self.x_label = r"$ \text{Time} / s$"
        self.y_label = r"$ \text{MSD} / m^{2}$"
        self.result_keys = ["diffusion_coefficient", "uncertainty"]
        self.result_series_keys = ["time", "msd"]
        self.analysis_name = "Einstein Self-Diffusion Coefficients"
        self.loop_condition = False
        self.density_array = None  # define empty msd array
        self.tau_values = None
        self.species = list()
        log.info("starting Particle Density Computation")

    @call
    def __call__(
        self,
        plot: bool = False,
        species: list = None,
        data_range: int = 100,
        number_of_configurations: int = 100,
        number_of_bins: int = 100,
        save: bool = True,
        optimize: bool = False,
        correlation_time: int = 1,
        atom_selection: np.s_ = np.s_[:],
        molecules: bool = False,
        tau_values: Union[int, List, Any] = np.s_[:],
        gpu: bool = False,
    ) -> None:
        """

        Parameters
        ----------
        plot : bool
                if true, plot the output.
        species : list
                List of species on which to operate.
        number_of_configurations : int
                Number of configurations to use in the density plot.
        number_of_bins : int
                Number of bins to use in the histogram.
        data_range : int
                Data range to use in the analysis.
        save : bool
                if true, save the output.
        optimize : bool
                If true, an optimization loop will be run.
        correlation_time : int
                Correlation time to use in the window sampling.
        atom_selection : np.s_
                Selection of atoms to use within the HDF5 database.
        molecules : bool
                If true, molecules are used instead of atoms.
        tau_values : Union[int, list, np.s_]
                Selection of tau values to use in the window sliding.
        gpu : bool
                If true, scale the memory requirement down to the amount of
                the biggest GPU in the system.

        Returns
        -------
        None
        """
        self.update_user_args(
            plot=plot,
            data_range=data_range,
            save=save,
            correlation_time=correlation_time,
            atom_selection=atom_selection,
            tau_values=tau_values,
            gpu=gpu,
        )

        self.species = species
        self.molecules = molecules
        self.number_of_configurations = number_of_configurations
        self.number_of_bins = number_of_bins
        edge_val = max(np.sqrt(np.array(self.experiment.box_array)**2)) / 2
        self.edges = np.linspace(-edge_val, edge_val, number_of_bins + 1)

        # attributes based on user args
        self.density_array = np.zeros((number_of_bins, 3))  # define empty msd array

        if species is None:
            if molecules:
                self.species = list(self.experiment.molecules)
            else:
                self.species = list(self.experiment.species)

        return self.update_db_entry_with_kwargs(
            data_range=data_range,
            correlation_time=correlation_time,
            molecules=molecules,
            atom_selection=atom_selection,
            tau_values=tau_values,
            species=species,
        )

    def _update_output_signatures(self):
        """
        After having run prepare_managers, update the output signatures.

        Returns
        -------
        Update the class state.
        """
        self.batch_output_signature = tf.TensorSpec(
            shape=(None, self.batch_size, 3), dtype=tf.float64
        )
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
        self.prefactor = 1

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
        data = tf.reduce_mean(tfp.stats.histogram(ensemble, self.edges, axis=1), axis=1)

        self.density_array += np.array(np.array(data))  # Update the averaged function

    def _run_visualization(self, species: str):
        """
        Run the visualizer.

        Parameters
        ----------
        species : str
                Name of the species being visualized.

        """
        visualizer = DataVisualizer3D(
            data=self.density_array, title=f"{species}-Density"
        )
        visualizer.plot()

    def _post_operation_processes(self, species: str = None):
        """
        Apply post-op processes such as saving and plotting.

        Returns
        -------

        """
        self._run_visualization(species)
