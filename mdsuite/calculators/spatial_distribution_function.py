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

        self.scale_function = {"quadratic": {"outer_scale_factor": 1}}
        self.loaded_property = "Positions"
        self.database_group = "Spatial_Distribution_Function"
        self.x_label = r"$$\text{r} /  \AA$$"  # None
        self.y_label = r"$$\text{g(r)}$$"  # None
        self.analysis_name = "Spatial_Distribution_Function"
        self.experimental = True

        self._dtype = tf.float32

    @call
    def __call__(
        self,
        molecules: bool = False,
        start: int = 1,
        stop: int = 10,
        number_of_configurations: int = 5,
        r_min: float = 4.0,
        r_max: float = 4.5,
        species: list = None,
        **kwargs,
    ):
        """
        User Interface to the Spatial Distribution Function

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
        self.sample_configurations = np.linspace(
            start, stop, number_of_configurations, dtype=np.int
        )

        self.update_user_args()
        self._check_input()

    def _load_positions(self, indices: list, species: str) -> tf.Tensor:
        """
        Load the positions matrix

        This function is here to optimize calculation speed

        Parameters
        ----------
        indices : list
                List of indices to take from the database_path
        species: str
                The species to load the positions from
        Returns
        -------
        loaded_data : tf.Tensor
                tf.Tensor of tensor_values loaded from the hdf5 database_path
        """
        # path_list = [join_path(species, "Positions") for species in self.species]

        path_list = [join_path(species, "Positions")]

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

        for idx, sample_configuration in tqdm(
            enumerate(np.array_split(self.sample_configurations, self.n_batches)),
            ncols=70,
        ):
            positions_tensor = []
            species_length = []
            for species in self.species:
                positions_tensor.append(
                    self._load_positions(sample_configuration, species)
                )
                species_length.append(len(positions_tensor[-1]))
                log.debug(f"Got {species_length[-1]} ions of {species}")

            positions_tensor = tf.concat(positions_tensor, axis=0)

            # make it (configurations, n_atoms, 3)
            positions_tensor = tf.transpose(positions_tensor, perm=(1, 0, 2))
            cell = tf.linalg.set_diag(tf.zeros((3, 3)), self.experiment.box_array)
            cell = tf.repeat(cell[None], positions_tensor.shape[0], axis=0)

            r_ij = nllayer({"positions": positions_tensor, "cell": cell})

            d_ij = tf.linalg.norm(r_ij, axis=-1)  # shape (b, i, j)
            # apply minimal and maximal distance and remove the diagonal elements of 0
            mask = (d_ij > self.r_min) & (d_ij < self.r_max)  # & (d_ij != 0)

            # Slicing the mask to the area where only the distances i!=j occur.
            # There are two such areas, so I am slicing them twice
            # could also mirror them
            mask_ = mask[:, species_length[0] :, : species_length[1]]
            r_ij_cut = r_ij[:, species_length[0] :, : species_length[1], :]
            r_ij_cut = r_ij_cut[mask_]
            sdf_values.append(r_ij_cut)
            # and the other half (only effective if species[0] != species[1])
            mask_ = mask[:, : species_length[0], species_length[1] :]
            r_ij_cut = r_ij[:, : species_length[0], species_length[1] :, :]
            r_ij_cut = r_ij_cut[mask_]
            sdf_values.append(r_ij_cut)

        sdf_values = tf.concat(sdf_values, axis=0)

        self._run_visualization(sdf_values)

    def _run_visualization(self, plot_data: tf.Tensor):
        """
        Run the visualizer.
        """
        visualizer = DataVisualizer3D(data=plot_data.numpy(), title="test")
        visualizer.plot()
