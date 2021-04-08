"""
Python module to calculate the translational dipole in a experiment.
"""

import numpy as np
from tqdm import tqdm
import os
import tensorflow as tf

from mdsuite.transformations.transformations import Transformations
from mdsuite.database.database import Database
from mdsuite.utils.meta_functions import join_path
from mdsuite.memory_management.memory_manager import MemoryManager
from mdsuite.database.data_manager import DataManager


class TranslationalDipoleMoment(Transformations):
    """
    Class to generate and store the ionic current of a experiment

    Attributes
    ----------
    experiment : object
            Experiment this transformation is attached to.
    """

    def __init__(self, experiment: object):
        """
        Constructor for the Ionic current calculator.

        Parameters
        ----------
        experiment : object
                Experiment this transformation is attached to.
        """
        super().__init__()
        self.experiment = experiment
        self.batch_size: int
        self.n_batches: int
        self.remainder: int

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

    def _check_for_charges(self):
        """
        Check the database_path for indices

        Returns
        -------

        """
        truth_table = []
        for item in self.experiment.species:
            path = join_path(item, 'Charge')
            truth_table.append(self.database.check_existence(path))

        if not all(truth_table):
            return False
        else:
            return True

    def _transformation(self, data: tf.Tensor):
        """
        Calculate the translational dipole moment of the system.

        Returns
        -------

        """
        positions_keys = []
        charge_keys = []
        for item in data:
            if str.encode('Unwrapped_Positions') in item:
                positions_keys.append(item)
            elif str.encode('Charge') in item:
                charge_keys.append(item)

        if len(charge_keys) != len(positions_keys):
            charges = False
        else:
            charges = True

        dipole_moment = tf.zeros(shape=(data[str.encode('data_size')], 3), dtype=tf.float64)
        if charges:
            for position, charge in zip(positions_keys, charge_keys):
                dipole_moment += tf.reduce_sum(data[position]*data[charge], axis=0)
        else:
            for item in positions_keys:
                species_string = item.decode("utf-8")
                species = species_string.split('/')[0]
                # Build the charge tensor for assignment
                charge = self.experiment.species[species]['charge'][0]
                charge_tensor = tf.ones(shape=(data[str.encode('data_size')], 3), dtype=tf.float64) * charge
                dipole_moment += tf.reduce_sum(data[item]*charge_tensor, axis=0)  # Calculate the final dipole moments

        return dipole_moment

    def _update_species_type_dict(self, dictionary: dict, path_list: list, dimension: int):
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
            species = item.split('/')[0]
            n_atoms = len(self.experiment.species[species]['indices'])
            dictionary[str.encode(item)] = tf.TensorSpec(shape=(n_atoms, None, dimension), dtype=tf.float64)

        return dictionary

    def _prepare_database_entry(self):
        """
        Add the relevant tensor_values sets and groups in the database_path

        Parameters
        ----------
        species : str
                Species for which tensor_values will be added.
        Returns
        -------
        tensor_values structure for use in saving the tensor_values to the database_path.
        """

        path = join_path('Translational_Dipole_Moment', 'Translational_Dipole_Moment')  # name of the new database_path
        dataset_structure = {path: (self.experiment.number_of_configurations, 3)}
        self.database.add_dataset(dataset_structure)  # add a new dataset to the database_path
        data_structure = {path: {'indices': np.s_[:], 'columns': [0, 1, 2]}}

        return data_structure

    def _compute_dipole_moment(self):
        """
        Loop over batches and compute the dipole moment
        Returns
        -------

        """
        type_spec = {}
        data_structure = self._prepare_database_entry()
        positions_path = [join_path(species, 'Unwrapped_Positions') for species in self.experiment.species]

        if self._check_for_charges():
            charge_path = [join_path(species, 'Charge') for species in self.experiment.species]
            data_path = np.concatenate((positions_path, charge_path))
            self._prepare_monitors(data_path)
            type_spec = self._update_species_type_dict(type_spec, positions_path, 3)
            type_spec = self._update_species_type_dict(type_spec, charge_path, 1)
        else:
            data_path = positions_path
            self._prepare_monitors(data_path)
            type_spec = self._update_species_type_dict(type_spec, positions_path, 3)
        type_spec[str.encode('data_size')] = tf.TensorSpec(None, dtype=tf.int16)
        batch_generator, batch_generator_args = self.data_manager.batch_generator(dictionary=True, remainder=True)
        data_set = tf.data.Dataset.from_generator(batch_generator,
                                                  args=batch_generator_args,
                                                  output_signature=type_spec)
        data_set.prefetch(tf.data.experimental.AUTOTUNE)
        for index, x in enumerate(data_set):
            data = self._transformation(x)
            self._save_coordinates(data, index, x[str.encode('data_size')], data_structure)

    def _save_coordinates(self, data: tf.Tensor, index: int, batch_size: int, data_structure: dict):
        """
        Save the tensor_values into the database_path

        Parameters
        ----------
        species : str
                name of species to update in the database_path.
        Returns
        -------
        saves the tensor_values to the database_path.
        """
        self.database.add_data(data=data,
                               structure=data_structure,
                               start_index=index,
                               batch_size=batch_size,
                               system_tensor=True)

    def run_transformation(self):
        """
        Run the ionic current transformation
        Returns
        -------

        """
        self._compute_dipole_moment()  # run the transformation.
        self.experiment.memory_requirements = self.database.get_memory_information()
