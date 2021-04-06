"""
Module for mapping atoms to molecules
"""

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from mdsuite.graph_modules.molecular_graph import MolecularGraph
from mdsuite.database.data_manager import DataManager
from mdsuite.memory_management.memory_manager import MemoryManager
from mdsuite.database.database import Database
from mdsuite.utils.meta_functions import join_path


class MolecularMap:
    """
    Class for mapping atoms in a database to molecules.
    """

    def __init__(self, experiment: object, molecules: dict):
        """
        Constructor for the MolecularMap class.

        Parameters
        ----------
        experiment : object
                Experiment object to work within.
        molecules : dict
                Molecule dictionary to use as reference. e.g. {'emim': {'smiles': 'CCN1C=C[N+](+C1)C', 'amount': 20},
                                                                'PF6': {'smiles': 'F[P-](F)(F)(F)(F)F', 'amount': 20}}
                would be the input for the emim-PF6 ionic liquid.
        """
        self.experiment = experiment
        self.molecules = molecules

        self.reference_molecules = {}
        self.adjacency_graphs = {}
        self.database = Database(name=os.path.join(self.experiment.database_path, "database.hdf5"),
                                 architecture='simulation')

    def _prepare_monitors(self, data_path: list):
        """
        Prepare the tensor_values and memory managers.

        Parameters
        ----------
        data_path : list
                List of tensor_values paths to load from the hdf5 database_path.

        Returns
        -------

        """
        self.memory_manager = MemoryManager(data_path=data_path, database=self.database, scaling_factor=5,
                                            memory_fraction=0.5)
        self.data_manager = DataManager(data_path=data_path, database=self.database)
        self.batch_size, self.n_batches, self.remainder = self.memory_manager.get_batch_size()
        self.data_manager.batch_size = self.batch_size
        self.data_manager.n_batches = self.n_batches
        self.data_manager.remainder = self.remainder

    def _prepare_database_entry(self, species, number_of_molecules: int):
        """
        Call some housekeeping methods and prepare for the transformations.
        Returns
        -------

        """
        # collect machine properties and determine batch size
        path = join_path(species, 'Unwrapped_Positions')  # name of the new database_path
        dataset_structure = {path: (number_of_molecules, self.experiment.number_of_configurations, 3)}
        self.database.add_dataset(dataset_structure)  # add a new dataset to the database_path
        data_structure = {path: {'indices': np.s_[:], 'columns': [0, 1, 2]}}

        return data_structure

    def _build_reference_graphs(self):
        """
        Build the reference graphs from the SMILES strings.
        Returns
        -------
        """
        for item in self.molecules:
            self.reference_molecules[item] = {}
            mol, species = MolecularGraph(self.experiment,
                                          from_smiles=True,
                                          smiles_string=self.molecules[item]['smiles']).build_smiles_graph()
            self.reference_molecules[item]['species'] = list(species)
            self.reference_molecules[item]['graph'] = mol
            self.reference_molecules[item]['mass'] = self._get_molecular_mass(species)

    def _get_molecular_mass(self, species_dict: dict):
        """
        Get the mass of a SMILES molecule based on experiment data.

        Parameters
        ----------
        species_dict : dict
                Dictionary of species information along with number of species present in the system.
        Returns
        -------
        mass : float
                mass of the molecule
        """
        mass = 0.0
        for item in species_dict:
            mass += self.experiment.species[item]['mass'] * species_dict[item]

        return mass

    def _build_configuration_graphs(self, cutoff: float = 1.9):
        """
        Build the adjacency graphs for the configurations
        Parameters
        ----------
        cutoff : float
                Custom cutoff parameter for bond length.
        Returns
        -------

        """
        for item in self.reference_molecules:
            self.adjacency_graphs[item] = {}
            mol = MolecularGraph(self.experiment,
                                 from_configuration=True,
                                 species=self.reference_molecules[item]['species'])
            self.adjacency_graphs[item]['graph'] = mol.build_configuration_graph(cutoff=cutoff)

    def _get_molecule_indices(self):
        """
        Collect the indices of the molecules.

        Returns
        -------

        """
        for item in self.adjacency_graphs:
            mol = MolecularGraph(self.experiment,
                                 from_configuration=True,
                                 species=self.reference_molecules[item]['species'])
            self.adjacency_graphs[item]['molecules'] = mol.reduce_graphs(self.adjacency_graphs[item]['graph'])

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
            dictionary[str.encode(item)] = tf.TensorSpec(shape=(None, self.batch_size, dimension), dtype=tf.float64)

        return dictionary

    def _load_batch(self, path_list: list, slice: np.s_, factor: list):
        """
        Load a batch of data and stack the configurations.
        Returns
        -------
        data : tf.Tensor
                A tensor of stacked data.
        """
        data = self.database.load_data(path_list=path_list, select_slice=slice, scaling=factor)
        return tf.concat(data, axis=0)

    def _prepare_mass_array(self, species: list):
        """
        Prepare an array of atom masses for the scaling.

        Parameters
        ----------
        species : list
                List of species to be loaded.
        Returns
        -------
        """
        mass_array = []
        for item in species:
            mass_array.append(self.experiment.species[item]['mass'])

        return mass_array

    def _map_molecules(self):
        """
        Map the molecules and save the data in the database.

        Returns
        -------

        """
        for item in self.molecules:
            species = self.reference_molecules[item]['species']
            mass_factor = self._prepare_mass_array(species)
            data_structure = self._prepare_database_entry(item, len(self.adjacency_graphs[item]['molecules']))
            path_list = [join_path(s, 'Unwrapped_Positions') for s in species]
            self._prepare_monitors(data_path=path_list)
            scaling_factor = self.reference_molecules[item]['mass']

            for i in tqdm(range(self.n_batches), ncols=70):
                start = i * self.batch_size
                stop = start + self.batch_size
                data = self._load_batch(path_list, np.s_[:, start:stop], factor=np.array(mass_factor) / scaling_factor)
                trajectory = np.zeros(shape=(len(self.adjacency_graphs[item]['molecules']), self.batch_size, 3))
                for t, molecule in enumerate(self.adjacency_graphs[item]['molecules']):
                    indices = list(self.adjacency_graphs[item]['molecules'][molecule].numpy())
                    trajectory[t, :, :] = np.sum(np.array(data)[indices], axis=0)
                self.database.add_data(data=trajectory,
                                       structure=data_structure,
                                       start_index=start,
                                       batch_size=self.batch_size,
                                       tensor=True)
            self.experiment.species[item] = {}
            self.experiment.species[item]['indices'] = [i for i in range(len(self.adjacency_graphs[item]['molecules']))]
            self.experiment.species[item]['mass'] = scaling_factor

    def run_transformation(self):
        """
        Perform the transformation
        Returns
        -------
        Update the experiment database.
        """
        self._build_reference_graphs()
        self._build_configuration_graphs()
        self._get_molecule_indices()
        self._map_molecules()
        self.experiment.save_class()