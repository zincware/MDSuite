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

    def __init__(self, obj, n_batches=1, n_confs=50, r_cut: int = 6.0, start: int = 0, stop: int = 10000,
                 bins: int = 500):
        """Compute the Angular Distribution Function for all species combinations

        Parameters
        ----------
        obj
        n_batches
        n_confs
        """
        # TODO update docstring!
        super().__init__(obj)
        self.r_cut = r_cut
        self.start = start
        self.stop = stop
        self.n_confs = n_confs
        self.bins = bins
        self.bin_range = [0.0, 3.15]
        self.n_batches = n_batches  # memory management for all batches
        self.n_batches_triples = 100  # memory management for triples generation per batch.

    def _load_positions(self, indices):
        # TODO make parent class?
        return self.parent.load_matrix("Positions", select_slice=np.s_[:, indices], tensor=True)

    def run_analysis(self):
        """
        Perform the ADF analysis
        """
        sample_configs = np.linspace(self.start, self.stop, self.n_confs, dtype=np.int)

        species_indices = []
        start_index = 0
        stop_index = 0
        for species in self.parent.species:
            stop_index += len(self.parent.species.get(species).get('indices'))
            species_indices.append(
                (species, start_index, stop_index)
            )
            start_index = stop_index

        angles = {}

        for idx in tqdm(np.array_split(sample_configs, self.n_batches), ncols=70):
            positions = self._load_positions(idx)

            tmp = tf.concat(positions, axis=0)
            tmp = tf.transpose(tmp, (1, 0, 2))

            r_ij_flat = next(get_neighbour_list(tmp, cell=self.parent.box_array, batch_size=1))  # TODO batch_size!
            r_ij_indices = get_triu_indicies(self.parent.number_of_atoms)

            r_ij_mat = tf.scatter_nd(
                indices=tf.transpose(r_ij_indices),
                updates=tf.transpose(r_ij_flat, (1, 2, 0)),  # Shape is now (n_atoms, n_atoms, 3, n_timesteps)
                shape=(self.parent.number_of_atoms, self.parent.number_of_atoms, 3, len(idx))
            )
            r_ij_mat -= tf.transpose(r_ij_mat, (1, 0, 2, 3))
            r_ij_mat = tf.transpose(r_ij_mat, (3, 0, 1, 2))

            r_ijk_indices = get_triplets(r_ij_mat, r_cut=self.r_cut, n_atoms=self.parent.number_of_atoms,
                                         n_batches=self.n_batches_triples)

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
            fig, ax = plt.subplots()
            ax.plot(np.linspace(self.bin_range[0] * (180 / 3.14159), self.bin_range[1] * (180 / 3.14159), self.bins),
                    hist)
            ax.set_title(name)
            fig.show()
            # TODO fig.show() does not work?
            fig.savefig(f"{self.parent.figures_path}/adf_{name}.png")