"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.
"""

from abc import ABC

import logging

import tensorflow as tf
import itertools
import numpy as np
from tqdm import tqdm

from mdsuite.calculators.calculator import Calculator
from mdsuite.utils.neighbour_list import get_neighbour_list, get_triu_indicies, get_triplets
from mdsuite.utils.linalg import get_angles
import matplotlib.pyplot as plt


class AngularDistributionFunction(Calculator, ABC):
    """
    Compute the Angular Distribution Function for all species combinations

    Parameters
    ----------
    batch_size : int
        Number of batches, to split the configurations into.
    n_minibatches: int
        Number of minibatches for computing the angles to split each batch into
    n_confs: int
        Number of configurations to analyse
    r_cut: float
        cutoff radius for the ADF
    start: int
        Index of the first configuration
    stop: int
        Index of the last configuration
    bins: int
        bins for the ADF
    use_tf_function: bool, default False
        activate the tf.function decorator for the minibatches. Can speed up the calculation significantly, but
        may lead to excessive use of memory! During the first batch, this function will be traced. Tracing is slow,
        so this might only be useful for a larger number of batches.
    """

    def __init__(self, experiment, batch_size: int = 1, n_minibatches: int = 50, n_confs: int = 5,
                 r_cut: int = 6.0, start: int = 1, stop: int = None, bins: int = 500, use_tf_function: bool = False,
                 export: bool = False, molecules: bool = False, gpu: bool = False, plot: bool = True):
        """
        Compute the Angular Distribution Function for all species combinations

        Parameters
        ----------
        batch_size : int
            Number of batches, to split the configurations into.
        n_minibatches: int
            Number of minibatches for computing the angles to split each batch into
        n_confs: int
            Number of configurations to analyse
        r_cut: float
            cutoff radius for the ADF
        start: int
            Index of the first configuration
        stop: int
            Index of the last configuration
        bins: int
            bins for the ADF
        use_tf_function: bool, default False
            activate the tf.function decorator for the minibatches. Can speed up the calculation significantly, but
            may lead to excessive use of memory! During the first batch, this function will be traced. Tracing is slow,
            so this might only be useful for a larger number of batches.
        """
        super().__init__(experiment, data_range=1, export=export, gpu=gpu, plot=plot)
        self.scale_function = {'quadratic': {'outer_scale_factor': 10}}

        self.use_tf_function = use_tf_function
        self.loaded_property = 'Positions'
        self.r_cut = r_cut
        self.start = start
        self.molecules = molecules
        self.experimental = True
        if stop is None:
            self.stop = experiment.number_of_configurations - 1
        else:
            self.stop = stop
        self.n_confs = n_confs
        self.bins = bins
        self.bin_range = [0.0, 3.15]  # from 0 to pi -. ??? 3.1415 -> 3.15 = failed rounding. I rounding up here, to
        # get a nice plot. If I would go down it would cut off some values.
        self._batch_size = batch_size  # memory management for all batches
        # TODO _n_batches is used instead of n_batches because the memory management is not yet implemented correctly
        self.n_minibatches = n_minibatches  # memory management for triples generation per batch.

        self.analysis_name = "Angular_Distribution_Function"
        self.database_group = "Angular_Distribution_Function"
        self.x_label = r'Angle ($\theta$)'
        self.y_label = 'ADF /a.u.'

        self.log = logging.getLogger(__name__)

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

        if self.molecules:
            path_list = [join_path(species, "Positions") for species in self.experiment.molecules]
            return self.experiment.load_matrix("Positions", path=path_list, select_slice=np.s_[:, indices])
        else:
            return self.experiment.load_matrix("Positions", select_slice=np.s_[:, indices])

    def run_experimental_analysis(self):
        """
        Perform the ADF analysis
        """
        sample_configs = np.linspace(self.start, self.stop, self.n_confs, dtype=np.int)

        species_indices = []
        start_index = 0
        stop_index = 0
        for species in self.experiment.species:
            stop_index += len(self.experiment.species.get(species).get('indices'))
            species_indices.append(
                (species, start_index, stop_index)
            )
            start_index = stop_index

        angles = {}

        # TODO select when to enable tqdm of get_triplets

        if self.use_tf_function:
            @tf.function
            def _get_triplets(x):
                return get_triplets(x, r_cut=self.r_cut, n_atoms=self.experiment.number_of_atoms,
                                    n_batches=self.n_minibatches, disable_tqdm=True)
        else:
            def _get_triplets(x):
                return get_triplets(x, r_cut=self.r_cut, n_atoms=self.experiment.number_of_atoms,
                                    n_batches=self.n_minibatches, disable_tqdm=True)

        def generator():
            for idx in sample_configs:
                if len(self.experiment.species) == 1:
                    yield self._load_positions(idx)  # Load the batch of positions
                else:
                    yield tf.concat(self._load_positions(idx), axis=0)  # Load th
                # yield self._load_positions(idx)  # <- bottleneck, because loading single configurations

        dataset = tf.data.Dataset.from_generator(generator, output_signature=(
            tf.TensorSpec(shape=(None, None), dtype=tf.float32)
        ))

        dataset = dataset.batch(self._batch_size).prefetch(tf.data.AUTOTUNE)

        self.log.debug(f'batch_size: {self._batch_size}')

        for positions in tqdm(dataset, total=self.n_confs):
            timesteps, atoms, _ = tf.shape(positions)

            tmp = tf.concat(positions, axis=0)

            r_ij_flat = next(get_neighbour_list(tmp, cell=self.experiment.box_array, batch_size=1))  # TODO batch_size!
            r_ij_indices = get_triu_indicies(self.experiment.number_of_atoms)

            r_ij_mat = tf.scatter_nd(
                indices=tf.transpose(r_ij_indices),
                updates=tf.transpose(r_ij_flat, (1, 2, 0)),  # Shape is now (n_atoms, n_atoms, 3, n_timesteps)
                shape=(self.experiment.number_of_atoms, self.experiment.number_of_atoms, 3, timesteps)
            )
            r_ij_mat -= tf.transpose(r_ij_mat, (1, 0, 2, 3))
            r_ij_mat = tf.transpose(r_ij_mat, (3, 0, 1, 2))

            r_ijk_indices = _get_triplets(r_ij_mat)

            for species in itertools.combinations_with_replacement(species_indices, 3):
                # Select the part of the r_ijk indices, where the selected species triple occurs.
                (i_name, i_min, i_max), (j_name, j_min, j_max), (k_name, k_min, k_max) = species

                name = f"{i_name}-{j_name}-{k_name}"

                i_condition = tf.logical_and(r_ijk_indices[:, 1] >= i_min, r_ijk_indices[:, 1] < i_max)
                j_condition = tf.logical_and(r_ijk_indices[:, 2] >= j_min, r_ijk_indices[:, 2] < j_max)
                k_condition = tf.logical_and(r_ijk_indices[:, 3] >= k_min, r_ijk_indices[:, 3] < k_max)

                condition = tf.math.logical_and(
                    x=tf.math.logical_and(x=i_condition, y=j_condition),
                    y=k_condition
                )

                tmp = tf.gather_nd(r_ijk_indices, tf.where(condition))
                # Get the indices required.

                if angles.get(name) is not None:
                    angles.update({
                        name: angles.get(name) + tf.histogram_fixed_width(get_angles(r_ij_mat, tmp), self.bin_range,
                                                                          self.bins)
                    })
                else:
                    angles.update({
                        name: tf.histogram_fixed_width(get_angles(r_ij_mat, tmp), self.bin_range, self.bins)
                    })

        for species in itertools.combinations_with_replacement(species_indices, 3):
            name = f"{species[0][0]}-{species[1][0]}-{species[2][0]}"
            hist = angles.get(name)

            bin_range_to_angles = np.linspace(self.bin_range[0] * (180 / 3.14159),
                                              self.bin_range[1] * (180 / 3.14159),
                                              self.bins)

            self.data_range = self.n_confs
            if self.save:
                self._save_data(name=self._build_table_name(name),
                                data=self._build_pandas_dataframe(bin_range_to_angles, hist))
            if self.export:
                self._export_data(name=self._build_table_name(name),
                                  data=self._build_pandas_dataframe(bin_range_to_angles, hist))
            if self.plot:
                fig, ax = plt.subplots()
                ax.plot(bin_range_to_angles, hist, label=name)
                ax.set_title(f"{name} - Max: {bin_range_to_angles[tf.math.argmax(hist)]:.3f}° ")
                self._plot_fig(fig, ax, title=name)

    def _calculate_prefactor(self, species: str = None):
        """
        calculate the calculator pre-factor.

        Parameters
        ----------
        species : str
                Species property if required.
        Returns
        -------

        """
        raise NotImplementedError

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
