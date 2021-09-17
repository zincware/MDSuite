"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: A spatial distribution function calculator
"""
from __future__ import annotations

import logging

from mdsuite.calculators.calculator import Calculator, call
from mdsuite.utils.tensorflow.layers import NLLayer
from mdsuite.utils.meta_functions import join_path
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mdsuite.visualizer.d3_data_visualizer import DataVisualizer3D

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdsuite import Experiment

log = logging.getLogger(__name__)


class SpatialDistributionFunction(Calculator):
    """Spatial Distribution Function Calculator based on the r_ij matrix"""

    def __init__(
            self, experiment: Experiment, experiments=None, load_data: bool = False
    ):
        """Constructor of the SpatialDistributionFunction

        Parameters
        ----------
        experiment: Experiment
            managed by RunComputation
        experiments:
            list of Experiments, managed by RunComputation
        load_data: bool
            managed by RunComputation

        """
        super().__init__(experiment, experiments=experiments, load_data=load_data)

        self.scale_function = {'quadratic': {'outer_scale_factor': 1}}
        self.loaded_property = 'Positions'
        self.database_group = 'Spatial_Distribution_Function'
        self.x_label = r'r ($\AA$)'  # None
        self.y_label = 'g(r)'  # None
        self.analysis_name = 'Spatial_Distribution_Function'
        self.experimental = True

        self._dtype = tf.float32

    @call
    def __call__(
            self, molecules: bool = False, start: int = 1, stop: int = 10, number_of_configurations: int = 5,
                 r_min: float = 4.0, r_max: float = 4.5, species: list = None, **kwargs):
        """User Interface to the Spatial Distribution Function

        Parameters
        ----------
        molecules
        start: int
            Index of the first configuration
        stop: int
            Index of the last configuration
        number_of_configurations: int
            Number of configurations to sample between start and stop
        r_min: float
            Minimal distance for the SDF
        r_max: float
            Maximal distance for the SDF
        species: list
            List of species to use, for computing the SDF,
            if None a single SDF of all available species will be computed
        kwargs
        """
        self.species = species
        self.molecules = molecules
        self.r_min = r_min
        self.r_max = r_max

        # choose sampled configurations
        self.sample_configurations = np.linspace(start,
                                                 stop,
                                                 number_of_configurations,
                                                 dtype=np.int)

        self.update_user_args()
        self._check_input()

    def _load_positions(self, indices: list) -> tf.Tensor:
        """
        Load the positions matrix

        This function is here to optimize calculation speed

        Parameters
        ----------
        indices : list
                List of indices to take from the database_path
        Returns
        -------
        loaded_data : tf.Tensor
                tf.Tensor of tensor_values loaded from the hdf5 database_path
        """
        path_list = [join_path(species, "Positions") for species in self.species]
        data = self.experiment.load_matrix(
            "Positions", path=path_list, select_slice=np.s_[:, indices]
        )
        if len(self.species) == 1:
            return tf.cast(data, dtype=self.dtype)
        else:
            return tf.cast(tf.concat(data, axis=0), dtype=self.dtype)

    def _check_input(self):
        """Check and correct the user input"""
        if self.species is None:
            self.species = list(self.experiment.species)

    def run_experimental_analysis(self):
        """Run the computation"""

        # Iterate over batches

        sdf_values = []

        nllayer = NLLayer()

        for idx, sample_configuration in tqdm(enumerate(np.array_split(self.sample_configurations, self.n_batches)),
                                              ncols=70):
            positions_tensor = self._load_positions(sample_configuration)

            # make it (configurations, n_atoms, 3)
            positions_tensor = tf.transpose(positions_tensor, perm=(1, 0, 2))
            cell = tf.linalg.set_diag(tf.zeros((3, 3)), self.experiment.box_array, )
            cell = tf.repeat(cell[None], positions_tensor.shape[0], axis=0)
            # TODO slice r_ij that only the selected species distances are still available, e.g. for species1 != species2
            #  maybe use a dictionary in the end

            r_ij = nllayer({"positions": positions_tensor, "cell": cell})

            d_ij = tf.linalg.norm(r_ij, axis=-1)
            # apply minimal and maximal distance and remove the diagonal elements of 0
            mask = (d_ij > self.r_min) & (d_ij < self.r_max) & (d_ij != 0)
            r_ij_cut = r_ij[mask]

            sdf_values.append(r_ij_cut)

        sdf_values = tf.concat(sdf_values, axis=0)

        self._run_visualization(sdf_values)

    def _run_visualization(self, plot_data: tf.Tensor):
        """
        Run the visualizer.
        """
        visualizer = DataVisualizer3D(data=plot_data, title='test')

    # Calculator class methods required by the parent class -- All are empty.
    def _apply_operation(self, data, index):
        """
        Perform operation on an ensemble.

        Parameters
        ----------
        One tensor_values range of tensor_values to operate on.

        Returns
        -------

        """
        pass

    def _apply_averaging_factor(self):
        """
        Apply an averaging factor to the tensor_values.
        Returns
        -------
        averaged copy of the tensor_values
        """
        pass

    def _post_operation_processes(self, species: str = None):
        """
        call the post-op processes
        Returns
        -------

        """
        pass

    def _update_output_signatures(self):
        """
        After having run _prepare managers, update the output signatures.

        Returns
        -------

        """
        pass
