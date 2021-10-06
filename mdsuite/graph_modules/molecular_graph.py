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
"""
import os
import numpy as np
from pysmiles import read_smiles
from mdsuite.utils.meta_functions import join_path
from mdsuite.database.simulation_database import Database
from tqdm import tqdm
import tensorflow as tf


class MolecularGraph:
    """
    Class for building and studying molecular graphs.
    """

    def __init__(
        self,
        experiment: object,
        from_smiles: bool = False,
        from_configuration: bool = True,
        smiles_string: str = None,
        species: list = None,
    ):
        """
        Constructor for the MolecularGraph class.

        Parameters
        ----------
        experiment : object
                Experiment object from which to read.
        from_smiles : bool
                Build graphs from a smiles string.
        from_configuration : bool
                Build graphs from a configuration.
        smiles_string : str
                SMILES string to read
        species : list
                List of species to build a graph with.
        """
        self.experiment = experiment
        self.from_smiles = from_smiles
        self.from_configuration = from_configuration

        self.smiles_string = smiles_string
        self.species = species

        self.database = Database(
            name=os.path.join(self.experiment.database_path, "database.hdf5"),
            architecture="simulation",
        )

    def _perform_checks(self):
        """
        Check the input for inconsistency.

        Returns
        -------
        """
        pass

    @staticmethod
    def get_neighbour_list(positions: tf.Tensor, cell: list = None) -> tf.Tensor:
        """
        Generate the neighbour list

        Parameters
        ----------
        positions: tf.Tensor
            Tensor with shape (number_of_configurations, n_atoms, 3)
            representing the coordinates
        cell: list
            If periodic boundary conditions are used, please supply the cell
            dimensions, e.g. [13.97, 13.97, 13.97]. If the cell is provided
            minimum image convention will be applied!
        batch_size: int
            Has to be evenly divisible by the the number of configurations.

        Returns
        -------
        generator object which results all distances for the current batch of
        time steps

        To get the real r_ij matrix for one time_step you can use the following:
            r_ij_mat = np.zeros((n_atoms, n_atoms, 3))
            r_ij_mat[np.triu_indices(n_atoms, k = 1)] = get_neighbour_list(``*args``)
            r_ij_mat -= r_ij_mat.transpose(1, 0, 2)

        """
        r_ij_matrix = tf.reshape(positions, (1, len(positions), 3)) - tf.reshape(
            positions, (len(positions), 1, 3)
        )
        if cell:
            r_ij_matrix -= tf.math.rint(r_ij_matrix / cell) * cell
        return tf.norm(r_ij_matrix, ord="euclidean", axis=2)

    def build_smiles_graph(self):
        """
        Build molecular graphs from SMILES strings.
        """
        mol = read_smiles(self.smiles_string, explicit_hydrogen=True)
        data = mol.nodes
        species = {}
        for i in range(len(data)):
            item = data[i].get("element")
            if item in species:
                species[item] += 1
            else:
                species[item] = 0
                species[item] += 1

        return mol, species

    @staticmethod
    def _apply_system_cutoff(tensor: tf.Tensor, cutoff: float) -> tf.Tensor:
        """
        Enforce a cutoff on a tensor

        Parameters
        ----------
        tensor : tf.Tensor
        cutoff : float
        """

        cutoff_mask = tf.cast(
            tf.less(tensor, cutoff), dtype=tf.int16
        )  # Construct the mask

        return tf.linalg.set_diag(cutoff_mask, np.zeros(len(tensor)))

    def build_configuration_graph(self, cutoff: float, adjacency: bool = True):
        """
        Build a graph for the configuration.

        Parameters
        ----------
        cutoff : float
        adjacency : bool
                If true, the adjacent matrix is returned.
        Returns
        -------

        """
        path_list = [join_path(species, "Positions") for species in self.species]
        data_dict =  self.database.load_data(
            path_list=path_list, select_slice=np.s_[:, 0]
        )
        data = []
        for item in path_list:
            data.append(data_dict[item])
        configuration_tensor = tf.concat(data, axis=0)
        distance_matrix = self.get_neighbour_list(
            configuration_tensor, cell=self.experiment.box_array
        )

        return self._apply_system_cutoff(distance_matrix, cutoff)

    @staticmethod
    def reduce_graphs(adjacency_matrix: tf.Tensor, n_molecules: int = None):
        """
        Reduce an adjacency matrix into a linear combination of sub-matrices.

        Parameters
        ----------
        adjacency_matrix : tf.Tensor
                Adjacency tensor to reduce.
        n_molecules : int
                Number of molecules that should be found after the reduction.
                If a number is passed here and the reduced number if not equal
                to the argument, the kernel is exited by a raised error. If
                nothing is passed, no checks are performed.
        """

        def check_a_in_b(a, b):
            """Check if any value of a is in b

            Parameters
            ----------
            a: tf.Tensor
            b: tf.Tensor

            Returns
            -------
            bool

            """
            x = tf.unstack(a)
            for x1 in x:
                if tf.reduce_any(b == x1):
                    return True
            return False

        molecules = {}
        # TODO speed up
        for i in tqdm(range(len(adjacency_matrix)), desc="Building molecules"):
            indices = tf.where(adjacency_matrix[i])
            indices = tf.reshape(indices, (len(indices)))
            if len(molecules) == 0:
                molecule = 0
                molecules[molecule] = indices
            else:
                molecule = None
                for mol in molecules:
                    if check_a_in_b(indices, molecules[mol]):
                        molecule = mol
                        molecules[mol] = tf.concat([molecules[mol], indices], 0)
                        molecules[mol] = tf.unique(molecules[mol])[0]
                        break
                if molecule is None:
                    molecule = len(molecules)
                    molecules[molecule] = indices

        del_list = []
        for item in molecules:
            test_dict = molecules.copy()
            test_dict.pop(item)
            for reference in test_dict:
                if all(elem in test_dict[reference] for elem in molecules[item]):
                    del_list.append(item)

        for item in del_list:
            molecules.pop(item)

        if n_molecules is None:
            return molecules
        else:
            if len(molecules) != n_molecules:
                raise ValueError(
                    "Expected number of molecules does not "
                    "match the amount computed, please adjust"
                    "parameters."
                )
            else:
                return molecules
