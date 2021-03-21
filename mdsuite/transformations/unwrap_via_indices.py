"""
Unwrap a set of coordinates based on dumped indices.
"""

from mdsuite.transformations.transformations import Transformations
from mdsuite.database.database import Database
from mdsuite.database.data_manager import DataManager
from mdsuite.memory_management.memory_manager import MemoryManager
from mdsuite.utils.meta_functions import join_path

import os
import sys
import tensorflow as tf
import time
import numpy as np


class UnwrapViaIndices(Transformations):
    """ Class to unwrap coordinates based on dumped index values """

    def __init__(self, experiment: object, species: list = None):
        """
        Constructor for the Ionic current calculator.

        Parameters
        ----------
        experiment : object
                Experiment this transformation is attached to.
        species : list
                Species on which this transformation should be applied.
        box : list
                Box vectors to multiply the indices by
        """
        super().__init__()
        self.experiment = experiment
        self.species = species
        self.database = Database(name=os.path.join(self.experiment.database_path, "database_path.hdf5"),
                                 architecture='simulation')
        if self.species is None:
            self.species = list(self.experiment.species)

        self.data_manager: DataManager
        self.memory_manager: MemoryManager
        self.batch_size: int
        self.n_batches: int
        self.remainder: int

    def _check_for_indices(self):
        """
        Check the database_path for indices

        Returns
        -------

        """
        truth_table = []
        for item in self.species:
            path = join_path(item, 'Box_Images')
            truth_table.append(self.database.check_existence(path))

        if not all(truth_table):
            print("Indices were not included in the database_path generation. Please check your simulation files.")
            sys.exit(1)

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

    def _transformation(self, data: tf.Tensor):
        """
        Apply the transformation to a batch of tensor_values.

        Parameters
        ----------
        data : tf.Tensor
                Data on which to operate.

        Returns
        -------
        Scaled coordinates : tf.Tensor
                Coordinates scaled by the image number.
        """
        return data[0] + tf.math.multiply(data[1], self.experiment.box_array)

    def _save_unwrapped_coordinates(self, data: tf.Tensor, index: int, batch_size: int, data_structure: dict):
        """
        Save the tensor_values into the database_path

        Parameters
        ----------
        data : tf.Tensor
                Tensor to save in the database_path
        index : int
                Index to start at in the database_path
        batch_size : int
                Size of each batch
        data_structure : dict
                Data structure to direct saving.
        Returns
        -------
        saves the tensor_values to the database_path.
        """
        self.database.add_data(data=data,
                               structure=data_structure,
                               start_index=index,
                               batch_size=batch_size,
                               tensor=True)

    def _prepare_database_entry(self, species: str):
        """
        Add the relevant datasets and groups in the database_path

        Parameters
        ----------
        species : str
                Species for which tensor_values will be added.
        Returns
        -------
        tensor_values structure for use in saving the tensor_values to the database_path.
        """
        path = join_path(species, 'Unwrapped_Positions')
        species_length = len(self.experiment.species[species]['indices'])
        number_of_configurations = self.experiment.number_of_configurations
        dataset_structure = {path: (species_length, number_of_configurations, 3)}
        self.database.add_dataset(dataset_structure)
        data_structure = {path: {'indices': np.s_[:], 'columns': [0, 1, 2], 'length': species_length}}

        return data_structure

    def _unwrap_particles(self):
        """
        Perform the unwrapping
        Returns
        -------
        Updates the database_path object.
        """
        for species in self.species:
            data_structure = self._prepare_database_entry(species)
            data_path = [join_path(species, 'Positions'), join_path(species, 'Box_Images')]
            self._prepare_monitors(data_path)
            batch_generator, batch_generator_args = self.data_manager.batch_generator()
            data_set = tf.data.Dataset.from_generator(batch_generator,
                                                      args=batch_generator_args,
                                                      output_signature=tf.TensorSpec(shape=(2, None,
                                                                                            self.batch_size, 3),
                                                                                     dtype=tf.float64)
                                                      )
            data_set.prefetch(tf.data.experimental.AUTOTUNE)
            for index, batch in enumerate(data_set):
                data = self._transformation(batch)
                self._save_unwrapped_coordinates(data, index, self.batch_size, data_structure)

    def run_transformation(self):
        """
        Perform the transformation.
        """
        self._check_for_indices()
        self._unwrap_particles()  # run the transformation
