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
import logging
from typing import List

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from mdsuite.database.mdsuite_properties import mdsuite_properties
from mdsuite.graph_modules.molecular_graph import MolecularGraph
from mdsuite.transformations.transformations import Transformations
from mdsuite.utils.meta_functions import join_path

log = logging.getLogger(__name__)


class MolecularMap(Transformations):
    """
    Class for mapping atoms in a database to molecules.

    Attributes
    ----------
    scale_function : dict
            A dictionary referencing the memory/time scaling function of the
            transformation.
    molecules : dict
            Molecule dictionary to use as reference. e.g.

            .. code-block::

                {'emim': {'smiles': 'CCN1C=C[N+](+C1)C', 'amount': 20}, 'PF6':
                {'smiles': 'F[P-](F)(F)(F)(F)F', 'amount': 20}}

            would be the input for the emim-PF6 ionic liquid.
    """

    def __init__(self):
        """
        Constructor for the MolecularMap class.
        """
        super().__init__()
        self.molecules = None
        self.reference_molecules = {}
        self.adjacency_graphs = {}
        self.dependency = mdsuite_properties.unwrapped_positions
        self.scale_function = {"quadratic": {"outer_scale_factor": 5}}

    def _prepare_database_entry(self, species: str, number_of_molecules: int) -> dict:
        """
        Call some housekeeping methods and prepare for the transformations.

        Parameters
        ----------
        species
                Name of the species to be added
        number_of_molecules : int
                Number of molecules to be added to the database.
        Returns
        -------
        data_structure : dict
                A data structure for the incoming data.
        """
        # collect machine properties and determine batch size
        path = join_path(species, "Unwrapped_Positions")  # name of the new database_path
        dataset_structure = {
            path: (number_of_molecules, self.experiment.number_of_configurations, 3)
        }
        self.database.add_dataset(
            dataset_structure
        )  # add a new dataset to the database_path
        # data_structure = {path: {"indices": np.s_[:], "columns": [0, 1, 2]}}
        data_structure = {
            path: {
                "indices": [i for i in range(number_of_molecules)],
                "columns": [0, 1, 2],
            }
        }

        return data_structure

    def _run_dependency_check(self):
        """
        Check that dependencies are fulfilled.

        Returns
        -------
        Calls a resolve method if dependencies are not met.
        """
        for sp_name in self.experiment.species:
            path = join_path(sp_name, self.dependency.name)
            if not self.database.check_existence(path):
                self.get_prop_through_transformation(sp_name, self.dependency)

    def _update_type_dict(self, dictionary: dict, path_list: list, dimension: int):
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
                shape=(None, None, dimension), dtype=tf.float64
            )

        return dictionary

    def _update_species_type_dict(
        self, dictionary: dict, path_list: list, dimension: int
    ):
        """
        Update a type spec dictionary for a species input.

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
            species = item.split("/")[0]
            n_atoms = self.experiment.species[species].n_particles
            dictionary[str.encode(item)] = tf.TensorSpec(
                shape=(n_atoms, None, dimension), dtype=tf.float64
            )

        return dictionary

    def _build_reference_graphs(self):
        """
        Build the reference graphs from the reference input.
        .
        Returns
        -------
        Updates the reference molecules attribute of the class. It should be populated
        after this call and look something like:

        {'bmim': {'species': ['C', 'H', 'N'], 'mass': 200, 'graph': tf.Tensor}}

        If a dict is used as a reference, the graph will not exist and stronger
        isomorphism checks will not be possible.
        """
        for item in self.molecules:
            self.reference_molecules[item] = {}
            if "smiles" in self.molecules[item]:
                mol, species = MolecularGraph(
                    self.experiment,
                    from_smiles=True,
                    smiles_string=self.molecules[item]["smiles"],
                ).build_smiles_graph()
                self.reference_molecules[item]["graph"] = mol
            elif "reference" in self.molecules[item]:
                species = self.molecules[item]["reference"]
            else:
                error_msg = (
                    "The minimum amount of data was not given to the mapping."
                    "Either provide a reference key with information about"
                    "Which species and the number of them are in the molecule,"
                    "or provide a SMILES string that can be used to compute "
                    "this information."
                )
                log.info(error_msg)
                raise ValueError("Provide either a reference key or smiles string")
            self.reference_molecules[item]["species"] = list(species)
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
                self.adjacency_graphs[item]["graph"],
                molecule_name=item,
                n_molecules=amount,
            )

    @staticmethod
    def _format_batch(batch: dict, scale_factor: np.array) -> tf.Tensor:
        """
        Reformat a batch of data into a tensor.

        Parameters
        ----------
        batch : dict
                Batch of data from the batch generator return as a dict. Each item in
                the dict will be taken and stacked into a tensor.
        scale_factor : np.array
                Factor by which to scale the coordinates as they are loaded. In this
                case, the reduced mass of each species in the molecule.

        Returns
        -------
        data : tf.Tensor
                Stacked data to be used in the mapping.
        """
        batch.pop(str.encode("data_size"))
        data = []
        for i, item in enumerate(batch):
            data.append(batch[item] * scale_factor[i])

        return tf.concat(data, 0)

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
        for molecule_name in self.molecules:
            species = self.reference_molecules[molecule_name]["species"]
            mass_factor = self._prepare_mass_array(species)
            data_structure = self._prepare_database_entry(
                molecule_name, len(self.adjacency_graphs[molecule_name]["molecules"])
            )
            path_list = [join_path(s, "Unwrapped_Positions") for s in species]
            self._prepare_monitors(data_path=path_list)
            scaling_factor = self.reference_molecules[molecule_name]["mass"]
            molecules = self.experiment.molecules
            molecules[molecule_name] = {}
            molecules[molecule_name]["n_particles"] = len(
                self.adjacency_graphs[molecule_name]["molecules"]
            )
            molecules[molecule_name]["indices"] = list(
                range(molecules[molecule_name]["n_particles"])
            )
            molecules[molecule_name]["mass"] = scaling_factor
            molecules[molecule_name]["groups"] = {}
            type_spec = {}
            for item in path_list:
                type_spec[str.encode(item)] = tf.TensorSpec(
                    shape=(None, None, 3), dtype=self.dtype
                )
            type_spec.update(
                {
                    str.encode("data_size"): tf.TensorSpec(shape=(), dtype=tf.int32),
                }
            )
            batch_generator, batch_generator_args = self.data_manager.batch_generator()
            type_spec.update(
                {
                    str.encode("data_size"): tf.TensorSpec(shape=(), dtype=tf.int32),
                }
            )
            data_set = tf.data.Dataset.from_generator(
                batch_generator, args=batch_generator_args, output_signature=type_spec
            )
            data_set = data_set.prefetch(tf.data.experimental.AUTOTUNE)

            for i, batch in tqdm(
                enumerate(data_set),
                ncols=70,
                total=self.n_batches,
                desc=f"Mapping molecule graphs onto trajectory for {molecule_name}",
            ):
                data = self._format_batch(
                    batch, scale_factor=np.array(mass_factor) / scaling_factor
                )
                batch_size = data.shape[1]
                trajectory = np.zeros(
                    shape=(
                        len(self.adjacency_graphs[molecule_name]["molecules"]),
                        batch_size,
                        3,
                    )
                )

                for t, molecule in enumerate(
                    self.adjacency_graphs[molecule_name]["molecules"]
                ):
                    # indices of the atoms inside the specific molecule.
                    indices = list(
                        self.adjacency_graphs[molecule_name]["molecules"][
                            molecule
                        ].numpy()
                    )
                    molecules[molecule_name]["groups"][t] = self._build_indices_dict(
                        indices, species
                    )
                    trajectory[t, :, :] = np.sum(np.array(data)[indices], axis=0)
                self._save_output(
                    data=trajectory,
                    data_structure=data_structure,
                    index=i * self.batch_size,
                )

            self.experiment.molecules = molecules
            # self.experiment.species.update(molecules)

    def _build_indices_dict(self, indices: List[int], species: List[str]) -> dict:
        """
        Build an indices dict to store the groups of atoms in each molecule.

        Parameters
        ----------
        indices : list[int]
                Indices of atoms belonging in the molecule
        species : list[str]
                Species in the molecules, in the order that they are loaded.
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
                    np.array(list(filter(lambda x: x < lengths[i], indices)))
                ).tolist()
            else:
                greater_array = list(filter(lambda x: x >= lengths[i - 1], indices))
                constrained_array = list(filter(lambda x: x < lengths[i], greater_array))
                indices_dict[item] = np.sort(
                    np.array(constrained_array) - (lengths[i - 1])
                ).tolist()

        return indices_dict

    def run_transformation(self, molecules: dict):
        """
        Perform the transformation.

        Parameters
        ----------
        molecules : dict
                Molecule dictionary to use as reference. The reference component is the
                most critical part. One can either use a smiles string or a reference
                dict as demonstrated below.

                e.g, the input for EMIM-PF6 ionic liquid would be:

                .. code-block::

                   {'emim': {'smiles': 'CCN1C=C[N+](+C1)C', 'amount': 20},
                   'PF6': {'smiles': 'F[P-](F)(F)(F)(F)F', 'amount': 20}}

                or:

                .. code-block::

                   {'emim': {'reference': {'C': , 'N': 'H': }}, 'amount': 20},
                   'PF6': {'reference': {'P': 1, 'F': 6}, 'amount': 20}}

        Returns
        -------
        Update the experiment database.
        """
        self.molecules = molecules
        self._run_dependency_check()
        self._build_reference_graphs()
        self._build_configuration_graphs()
        self._get_molecule_indices()
        self._map_molecules()
        self.experiment.run.CoordinateWrapper(species=[key for key in self.molecules])
