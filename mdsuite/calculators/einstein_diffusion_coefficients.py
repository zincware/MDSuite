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
Module for the computation of self-diffusion coefficients using the Einstein method.
"""
from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass
from typing import Any, List, Union

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from mdsuite.calculators.calculator import call
from mdsuite.calculators.trajectory_calculator import TrajectoryCalculator
from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.utils.calculator_helper_methods import fit_einstein_curve

log = logging.getLogger(__name__)


@dataclass
class Args:
    """
    Data class for the saved properties.
    """

    data_range: int
    correlation_time: int
    atom_selection: np.s_
    tau_values: np.s_
    molecules: bool
    species: list


class EinsteinDiffusionCoefficients(TrajectoryCalculator, ABC):
    """
    Class for the Einstein diffusion coefficient implementation

    Attributes
    ----------
    msd_array : np.ndarray
            MSd data updated during each ensemble computation.

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    project.experiment.run.EinsteinDiffusionCoefficients(data_range=500,
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
        """

        super().__init__(**kwargs)
        self.scale_function = {"linear": {"scale_factor": 150}}
        self.loaded_property = mdsuite_properties.unwrapped_positions
        self.x_label = r"$ \text{Time} / s$"
        self.y_label = r"$ \text{MSD} / m^{2}$"
        self.result_keys = ["diffusion_coefficient", "uncertainty"]
        self.result_series_keys = ["time", "msd"]
        self.analysis_name = "Einstein Self-Diffusion Coefficients"
        self._dtype = tf.float64

        self.msd_array = None

        log.info("starting Einstein Diffusion Computation")

    @call
    def __call__(
        self,
        plot: bool = True,
        species: list = None,
        data_range: int = 100,
        correlation_time: int = 1,
        atom_selection: np.s_ = np.s_[:],
        molecules: bool = False,
        tau_values: Union[int, List, Any] = np.s_[:],
    ):
        """

        Parameters
        ----------
        plot : bool
                if true, plot the output.
        species : list
                List of species on which to operate.
        data_range : int
                Data range to use in the analysis.
        correlation_time : int
                Correlation time to use in the window sampling.
        atom_selection : np.s_
                Selection of atoms to use within the HDF5 database.
        molecules : bool
                If true, molecules are used instead of atoms.
        tau_values : Union[int, list, np.s_]
                Selection of tau values to use in the window sliding.

        Returns
        -------
        None
        """
        if species is None:
            if molecules:
                species = list(self.experiment.molecules)
            else:
                species = list(self.experiment.species)
        # set args that will affect the computation result
        self.args = Args(
            data_range=data_range,
            correlation_time=correlation_time,
            atom_selection=atom_selection,
            tau_values=tau_values,
            molecules=molecules,
            species=species,
        )
        self.plot = plot
        self.system_property = False
        self.time = self._handle_tau_values()
        self.msd_array = np.zeros(self.data_resolution)  # define empty msd array

    def check_input(self):
        """
        Check the user input to ensure no conflicts are present.

        Returns
        -------

        """
        self._run_dependency_check()

    def calculate_prefactor(self, species: str = None):
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
        if self.args.molecules:
            # Calculate the prefactor
            numerator = self.experiment.units["length"] ** 2
            denominator = (
                self.experiment.units["time"]
                * len(self.experiment.molecules[species]["indices"])
            ) * 6
        else:
            # Calculate the prefactor
            numerator = self.experiment.units["length"] ** 2
            denominator = (
                self.experiment.units["time"]
                * self.experiment.species[species].n_particles
            ) * 6

        self.prefactor = numerator / denominator

    def msd_operation(self, ensemble: tf.Tensor, square: bool = True):
        """
        Perform a simple msd operation.

        Parameters
        ----------
        ensemble : tf.Tensor
            Trajectory over which to compute the msd.
        square : bool
            If true, square the result, else just return the difference.
        Returns
        -------
        msd : tf.Tensor shape = (n_atoms, data_range, 3)
                Mean square displacement.
        """
        if square:
            return tf.math.squared_difference(
                tf.gather(ensemble, self.args.tau_values, axis=1), ensemble[:, None, 0]
            )
        else:
            return tf.math.subtract(ensemble, ensemble[:, None, 0])

    def ensemble_operation(self, ensemble):
        """
        Calculate and return the msd.

        Parameters
        ----------
        ensemble : tf.Tensor
                An ensemble of data to be operated on.

        Returns
        -------
        MSD of the tensor_values.
        """
        msd = self.msd_operation(ensemble)

        # Sum over trajectory and then coordinates and apply averaging and pre-factors
        msd = self.prefactor * tf.reduce_sum(tf.reduce_sum(msd, axis=0), axis=1)
        self.msd_array += np.array(msd)  # Update the averaged function

    def postprocessing(self, species: str):
        """
        Run post-processing on the data.

        This will include an averaging factor, saving the results, and plotting
        the msd against time.

        Parameters
        ----------
        species : str
                Name of the species being studied.

        Returns
        -------
        This method will scale the data by the number of ensembles, compute the
        diffusion coefficient, and store all of the results in the SQL database.
        """
        self.msd_array /= int(self.n_batches) * self.ensemble_loop

        result = fit_einstein_curve([self.time, self.msd_array])
        log.debug(f"Saving {species}")

        data = {
            self.result_keys[0]: result[0],
            self.result_keys[1]: result[1],
            self.result_series_keys[0]: self.time.tolist(),
            self.result_series_keys[1]: self.msd_array.tolist(),
        }

        self.queue_data(data=data, subjects=[species])

    def run_calculator(self):
        """
        Run analysis.

        Returns
        -------

        """
        self.check_input()
        # Loop over species
        for species in self.args.species:
            dict_ref = str.encode("/".join([species, self.loaded_property.name]))
            self.calculate_prefactor(species)

            batch_ds = self.get_batch_dataset([species])

            for batch in tqdm(
                batch_ds,
                ncols=70,
                desc=species,
                total=self.n_batches,
                disable=self.memory_manager.minibatch,
            ):
                ensemble_ds = self.get_ensemble_dataset(batch, species)

                for ensemble in ensemble_ds:
                    self.ensemble_operation(ensemble[dict_ref])

            # Scale, save, and plot the data.
            self.postprocessing(species)
