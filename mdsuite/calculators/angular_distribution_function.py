from abc import ABC

import tensorflow as tf
import itertools
import numpy as np
from tqdm import tqdm

from mdsuite.calculators.calculator import Calculator
from mdsuite.utils.neighbour_list import get_neighbour_list, get_triu_indicies, get_triplets
from mdsuite.utils.linalg import get_angles
import matplotlib.pyplot as plt


class AngularDistributionFunction(Calculator, ABC):

    def __init__(self, experiment, n_batches: int = 1, n_minibatches: int = 50, n_confs: int = 50, data_range=1,
                 r_cut: int = 6.0, start: int = 0, stop: int = 10000, bins: int = 500, use_tf_function: bool = False):
        """
        Compute the Angular Distribution Function for all species combinations

        Parameters
        ----------
        n_batches : int
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
        super().__init__(experiment, data_range=data_range)
        self.use_tf_function = use_tf_function
        self.loaded_property = 'Positions'
        self.r_cut = r_cut
        self.start = start
        self.experimental = True
        self.stop = stop
        self.n_confs = n_confs
        self.bins = bins
        self.bin_range = [0.0, 3.15]  # from 0 to pi -. ??? 3.1415 -> 3.15 = failed rounding.
        self.n_batches = n_batches  # memory management for all batches
        self.n_minibatches = n_minibatches  # memory management for triples generation per batch.

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

        if self.use_tf_function:
            @tf.function
            def _get_triplets(x):
                return get_triplets(x, r_cut=self.r_cut,
                                    n_atoms=self.experiment.number_of_atoms, n_batches=self.n_minibatches)
        else:
            def _get_triplets(x):
                return get_triplets(x, r_cut=self.r_cut,
                                    n_atoms=self.experiment.number_of_atoms, n_batches=self.n_minibatches)

        for idx in tqdm(np.array_split(sample_configs, self.n_batches), ncols=70):
            positions = self._load_positions(idx)

            tmp = tf.concat(positions, axis=0)
            tmp = tf.transpose(tmp, (1, 0, 2))

            r_ij_flat = next(get_neighbour_list(tmp, cell=self.experiment.box_array, batch_size=1))  # TODO batch_size!
            r_ij_indices = get_triu_indicies(self.experiment.number_of_atoms)

            r_ij_mat = tf.scatter_nd(
                indices=tf.transpose(r_ij_indices),
                updates=tf.transpose(r_ij_flat, (1, 2, 0)),  # Shape is now (n_atoms, n_atoms, 3, n_timesteps)
                shape=(self.experiment.number_of_atoms, self.experiment.number_of_atoms, 3, len(idx))
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

            fig, ax = plt.subplots()
            ax.plot(bin_range_to_angles, hist)
            ax.set_title(f"{name} - Max: {bin_range_to_angles[tf.math.argmax(hist)]:.3f}° ")
            fig.show()
            # fig.savefig(f"{self.experiment.figures_path}/adf_{name}.png")
            fig.savefig(fr"adf_{name}.png")
            np.save(f"{species}", (bin_range_to_angles, hist))

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
