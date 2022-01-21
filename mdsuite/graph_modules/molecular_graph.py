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
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf
from pysmiles import read_smiles
from tqdm import tqdm

from mdsuite.database.simulation_database import Database
from mdsuite.utils.meta_functions import join_path

if TYPE_CHECKING:
    from mdsuite.experiment import Experiment


class MolecularGraph:
    """
    Class for building and studying molecular graphs.
    """

    def __init__(
        self,
        experiment: Experiment,
        from_smiles: bool = False,
        from_configuration: bool = True,
        smiles_string: str = None,
        reference_dict: dict = None,
        species: list = None,
    ):
        """
        Constructor for the MolecularGraph class.

        Parameters
        ----------
        experiment : Experiment
                Experiment object from which to read.
        from_smiles : bool
                Build graphs from a smiles string.
        from_configuration : bool
                Build graphs from a configuration.
        smiles_string : str
                SMILES string to read and used in the construction of a reference
                graph with molecule information.
        reference_dict : dict
                Alternatively to the SMILES string a reference dict can be given to
                construct a reference molecule. This dict should be of the form:

                .. code-block: python

                   {'PF6': {'P': 1, 'F': 6}}}

                for a PF6 molecule.
        species : list
                List of species to build a graph with.
        """
        self.experiment = experiment
        self.from_smiles = from_smiles
        self.from_configuration = from_configuration

        self.smiles_string = smiles_string
        self.reference_dict = reference_dict
        self.species = species

        self.database = Database(self.experiment.database_path / "database.hdf5")

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

        Returns
        -------
        neighbour_list : tf.Tensor
                Neighbour list for a single configuration.

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
        Enforce a cutoff on a tensor.

        In this context the cutoff is used to identify bonded atoms. We argue
        that the closest atoms will be bonded as defined by a cutoff.
        Constructing the mask of closest atoms will allow for the bonded ones to be
        identified.

        Parameters
        ----------
        tensor : tf.Tensor
                Tensor of any size or shape to be masked. In our case it is a distance
                tensor of the atoms in a configuration.
        cutoff : float
                Cutoff to use in the mask. If a distance is greater than this cutoff it
                is marked as 0, if not, it is 1.
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
                Cutoff over which to look for molecules.
        adjacency : bool
                If true, the adjacency matrix is returned.
        Returns
        -------

        """
        path_list = [join_path(species, "Positions") for species in self.species]
        data_dict = self.database.load_data(path_list=path_list, select_slice=np.s_[:, 0])
        data = []
        for item in path_list:
            data.append(data_dict[item])
        configuration_tensor = tf.concat(data, axis=0)
        distance_matrix = self.get_neighbour_list(
            configuration_tensor, cell=self.experiment.box_array
        )

        return self._apply_system_cutoff(distance_matrix, cutoff)

    @staticmethod
    def reduce_graphs(
        adjacency_matrix: tf.Tensor, molecule_name: str, n_molecules: int = None
    ):
        """
        Reduce an adjacency matrix into a linear combination of sub-matrices.

        Parameters
        ----------
        adjacency_matrix : tf.Tensor
                Adjacency tensor to reduce.
        molecule_name : str
                Name of the molecule for better transparency during operation.
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
        for i in tqdm(
            range(len(adjacency_matrix)),
            desc=f"Building molecular graph from configuration for {molecule_name}",
        ):
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
                    f"Expected number of molecules ({n_molecules}) does not "
                    f"match the amount computed ({len(molecules)}), please adjust "
                    "parameters."
                )
            else:
                return molecules
