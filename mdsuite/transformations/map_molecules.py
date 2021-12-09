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
from typing import List

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from mdsuite.graph_modules.molecular_graph import MolecularGraph
from mdsuite.transformations.transformations import Transformations
from mdsuite.utils.meta_functions import join_path


class MolecularMap(Transformations):
    """
    Class for mapping atoms in a database to molecules.

    Attributes
    ----------
    scale_function : dict
            A dictionary referencing the memory/time scaling function of the
            transformation.
    experiment : object
            Experiment object to work within.
    molecules : dict
            Molecule dictionary to use as reference. e.g.

            .. code-block::

                {'emim': {'smiles': 'CCN1C=C[N+](+C1)C', 'amount': 20}, 'PF6':
                {'smiles': 'F[P-](F)(F)(F)(F)F', 'amount': 20}}

            would be the input for the emim-PF6 ionic liquid.
    """

    def __init__(self, molecules: dict):
        """Constructor for the MolecularMap class.

        Parameters
        ----------
        molecules : dict
                Molecule dictionary to use as reference. e.g, the input for
                emim-PF6 ionic liquid would be.

                .. code-block::

                   {'emim': {'smiles': 'CCN1C=C[N+](+C1)C', 'amount': 20},
                   'PF6': {'smiles': 'F[P-](F)(F)(F)(F)F', 'amount': 20}}

        """
        super().__init__()
        self.molecules = molecules
        self.reference_molecules = {}
        self.adjacency_graphs = {}
        self.dependency = "Unwrapped_Positions"
        self.scale_function = {"quadratic": {"outer_scale_factor": 5}}

    def _prepare_database_entry(self, species, number_of_molecules: int) -> dict:
        """
        Call some housekeeping methods and prepare for the transformations.
        Returns
        -------
        data_structure : dict
                A data structure for the incoming data.
        """
        # collect machine properties and determine batch size
        path = join_path(
            species, "Unwrapped_Positions"
        )  # name of the new database_path
        dataset_structure = {
            path: (number_of_molecules, self.experiment.number_of_configurations, 3)
        }
        self.database.add_dataset(
            dataset_structure
        )  # add a new dataset to the database_path
        # data_structure = {path: {"indices": np.s_[:], "columns": [0, 1, 2]}}
        data_structure = {
            path: {
                "indices": [i for i in range(number_of_molecules)], "columns": [0, 1, 2]
            }
        }

        return data_structure

    def _build_reference_graphs(self):
        """
        Build the reference graphs from the SMILES strings.
        Returns
        -------
        Nothing.
        """
        for item in self.molecules:
            self.reference_molecules[item] = {}
            mol, species = MolecularGraph(
                self.experiment,
                from_smiles=True,
                smiles_string=self.molecules[item]["smiles"],
            ).build_smiles_graph()
            self.reference_molecules[item]["species"] = list(species)
            self.reference_molecules[item]["graph"] = mol
            self.reference_molecules[item]["mass"] = self._get_molecular_mass(species)

    def _get_molecular_mass(self, species_dict: dict) -> float:
        """
        Get the mass of a SMILES molecule based on experiment data.

        Parameters
        ----------
        species_dict : dict
                Dictionary of species information along with number of species
                present in the system.
        Returns
        -------
        mass : float
                mass of the molecule
        """
        mass = 0.0
        for item in species_dict:
            mass += self.experiment.species[item]["mass"][0] * species_dict[item]

        return mass

    def _build_configuration_graphs(self):
        """
        Build the adjacency graphs for the configurations

        Returns
        -------
        Nothing.
        """
        for item in self.reference_molecules:
            self.adjacency_graphs[item] = {}
            mol = MolecularGraph(
                self.experiment,
                from_configuration=True,
                species=self.reference_molecules[item]["species"],
            )
            self.adjacency_graphs[item]["graph"] = mol.build_configuration_graph(
                cutoff=self.molecules[item]["cutoff"]
            )

    def _get_molecule_indices(self):
        """
        Collect the indices of the molecules.

        Returns
        -------
        Nothing.
        """
        for item in self.adjacency_graphs:
            try:
                amount = self.molecules[item]["amount"]
            except ValueError:
                amount = None
            mol = MolecularGraph(
                self.experiment,
                from_configuration=True,
                species=self.reference_molecules[item]["species"],
            )
            self.adjacency_graphs[item]["molecules"] = mol.reduce_graphs(
                self.adjacency_graphs[item]["graph"], n_molecules=amount
            )

    def _update_type_dict(
        self, dictionary: dict, path_list: list, dimension: int
    ) -> dict:
        """
        Update a type spec dictionary.

        Parameters
        ----------
        dictionary : dict
                Dictionary to append
        path_list : list
                List of paths for the dictionary
        dimension : int
                Dimension of the property
        Returns
        -------
        type dict : dict
                Dictionary for the type spec.
        """
        for item in path_list:
            dictionary[str.encode(item)] = tf.TensorSpec(
                shape=(None, self.batch_size, dimension), dtype=tf.float64
            )

        return dictionary

    def _load_batch(self, path_list: list, slice: np.s_, factor: list) -> tf.Tensor:
        """
        Load a batch of data and stack the configurations.
        Returns
        -------
        data : tf.Tensor
                A tensor of stacked data.
        """
        data_dict = self.database.load_data(
            path_list=path_list, select_slice=slice, scaling=factor, dictionary=True
        )
        data = []
        for item in path_list:
            data.append(data_dict[item])

        return tf.concat(data, axis=0)

    def _prepare_mass_array(self, species: list) -> list:
        """
        Prepare an array of atom masses for the scaling.

        Parameters
        ----------
        species : list
                List of species to be loaded.
        Returns
        -------
        mass_array : list
                A list of masses.
        """
        mass_array = []
        for item in species:
            mass_array.append(self.experiment.species[item]["mass"])

        return mass_array

    def _map_molecules(self):
        """
        Map the molecules and save the data in the database.

        Returns
        -------
        Updates the database.
        """
        for item in self.molecules:
            species = self.reference_molecules[item]["species"]
            mass_factor = self._prepare_mass_array(species)
            data_structure = self._prepare_database_entry(
                item, len(self.adjacency_graphs[item]["molecules"])
            )
            path_list = [join_path(s, "Unwrapped_Positions") for s in species]
            self._prepare_monitors(data_path=path_list)
            scaling_factor = self.reference_molecules[item]["mass"]
            molecules = self.experiment.molecules
            molecules[item] = {}
            molecules[item]["indices"] = [
                i for i in range(len(self.adjacency_graphs[item]["molecules"]))
            ]
            molecules[item]["mass"] = scaling_factor
            molecules[item]["groups"] = {}
            for i in tqdm(range(self.n_batches), ncols=70, desc="Mapping molecules"):
                start = i * self.batch_size
                stop = start + self.batch_size
                data = self._load_batch(
                    path_list,
                    np.s_[:, start:stop],
                    factor=np.array(mass_factor) / scaling_factor,
                )
                trajectory = np.zeros(
                    shape=(
                        len(self.adjacency_graphs[item]["molecules"]),
                        self.batch_size,
                        3,
                    )
                )

                for t, molecule in enumerate(self.adjacency_graphs[item]["molecules"]):
                    indices = list(
                        self.adjacency_graphs[item]["molecules"][molecule].numpy()
                    )
                    molecules[item]["groups"][t] = self._build_indices_dict(
                        indices, species
                    )
                    trajectory[t, :, :] = np.sum(np.array(data)[indices], axis=0)
                self._save_coordinates(
                    data=trajectory,
                    data_structure=data_structure,
                    index=start,
                    batch_size=self.batch_size,
                    system_tensor=False,
                    tensor=True,
                )
            self.experiment.molecules = molecules

    def _build_indices_dict(self, indices: List[int], species: List[str]) -> dict:
        """
        Build an indices dict to store the groups of atoms in each molecule.

        Parameters
        ----------
        indices : list[int]
                Indices of atoms belonging in the molecule
        species : list[str]
                Elements in the molecules, in the order that they are loaded.
        Returns
        -------
        group_dict : dict
                A dictionary of atoms and indices that specify that indices of
                this species is in a molecule.
        """
        indices_dict = {}
        lengths = []
        for i, item in enumerate(species):
            length = self.experiment.species[item].n_particles
            if i == 0:
                lengths.append(length)
            else:
                lengths.append(length + lengths[i - 1])

        for i, item in enumerate(species):
            if i == 0:
                indices_dict[item] = np.sort(
                    list(filter(lambda x: x < lengths[i], indices))
                ).tolist()
            else:
                greater_array = list(filter(lambda x: x >= lengths[i - 1], indices))
                constrained_array = list(
                    filter(lambda x: x < lengths[i], greater_array)
                )
                indices_dict[item] = np.sort(
                    np.array(constrained_array) - (lengths[i - 1] - 1)
                ).tolist()

        return indices_dict

    def run_transformation(self):
        """
        Perform the transformation
        Returns
        -------
        Update the experiment database.
        """
        self._run_dependency_check()
        self._build_reference_graphs()
        self._build_configuration_graphs()
        self._get_molecule_indices()
        self._map_molecules()
        self.experiment.perform_transformation(
            "WrapCoordinates", species=[item for item in self.molecules]
        )
