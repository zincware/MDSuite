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
        self.database = Database(name=os.path.join(self.experiment.database_path, "database.hdf5"),
                                 architecture='simulation')
        if self.species is None:
            self.species = list(self.experiment.species)

        self.data_manager: DataManager
        self.memory_manager: MemoryManager
        self.batch_size: int
        self.n_batches: int
        print(self.experiment.box_array)

    def _check_for_indices(self):
        """
        Check the database for indices

        Returns
        -------

        """
        truth_table = []
        for item in self.species:
            path = join_path(item, 'Box_Images')
            truth_table.append(self.database.check_existence(path))

        if not all(truth_table):
            print("Indices were not included in the database generation. Please check your simulation files.")
            sys.exit(1)

    def _prepare_monitors(self, data_path: list):
        """
        Prepare the data and memory managers.

        Parameters
        ----------
        data_path : list
                List of data paths to load from the hdf5 database.

        Returns
        -------

        """
        self.memory_manager = MemoryManager(data_path=data_path, database=self.database, scaling_factor=5,
                                            memory_fraction=0.5)
        self.data_manager = DataManager(data_path=data_path, database=self.database)
        self.batch_size, self.n_batches = self.memory_manager.get_batch_size()
        self.data_manager.batch_size = self.batch_size
        self.data_manager.batch_number = self.n_batches

    def _transformation(self, data: tf.Tensor):
        """
        Apply the transformation to a batch of data.

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
        Save the data into the database

        Parameters
        ----------
        data : tf.Tensor
                Tensor to save in the database
        index : int
                Index to start at in the database
        batch_size : int
                Size of each batch
        data_structure : dict
                Data structure to direct saving.
        Returns
        -------
        saves the data to the database.
        """
        self.database.add_data(data=data,
                               structure=data_structure,
                               database=self.database.open(),
                               start_index=index,
                               batch_size=batch_size,
                               tensor=True)

    def _prepare_database_entry(self, species: str):
        """
        Add the relevant datasets and groups in the database

        Parameters
        ----------
        species : str
                Species for which data will be added.
        Returns
        -------
        data structure for use in saving the data to the database.
        """
        path = join_path(species, 'Unwrapped_Positions')
        species_length = len(self.experiment.species[species]['indices'])
        number_of_configurations = self.experiment.number_of_configurations
        dataset_structure = {path: (species_length, number_of_configurations, 3)}
        self.database.add_dataset(dataset_structure, self.database.open())
        data_structure = {path: {'indices': np.s_[:], 'columns': [0, 1, 2], 'length': species_length}}

        return data_structure

    def _unwrap_particles(self):
        """
        Perform the unwrapping
        Returns
        -------
        Updates the database object.
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
            start = time.time()
            for index, x in enumerate(data_set):
                data = self._transformation(x)
                self._save_unwrapped_coordinates(data, index, self.batch_size, data_structure)

            print(time.time() - start)

    def run_transformation(self):
        """
        Perform the transformation.
        """
        self._check_for_indices()
        self._unwrap_particles()  # run the transformation
