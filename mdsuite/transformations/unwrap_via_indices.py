"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Unwrap a set of coordinates based on dumped indices.
"""
from mdsuite.transformations.transformations import Transformations
from mdsuite.utils.meta_functions import join_path
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm


class UnwrapViaIndices(Transformations):
    """
    Class to unwrap coordinates based on dumped index values

    Attributes
    ----------
    experiment : object
            Experiment this transformation is attached to.
    species : list
            Species on which this transformation should be applied.
    scale_function : dict
            A dictionary referencing the memory/time scaling function of the transformation.
    """

    def __init__(self, experiment: object, species: list = None):
        """
        Constructor for the Ionic current calculator.

        Parameters
        ----------
        experiment : object
                Experiment this transformation is attached to.
        species : list
                Species on which this transformation should be applied.
        """
        super().__init__(experiment)
        self.scale_function = {'linear': {'scale_factor': 2}}

        self.species = species
        if self.species is None:
            self.species = list(self.experiment.species)

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

    def _save_unwrapped_coordinates(self,
                                    data: tf.Tensor,
                                    index: int,
                                    batch_size: int,
                                    data_structure: dict):
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
        existing = self._run_dataset_check(path)
        if existing:
            old_shape = self.database.get_data_size(path)
            species_length = len(self.experiment.species[species]['indices'])
            resize_structure = {path: (species_length,
                                       self.experiment.number_of_configurations - old_shape[0], 3)}
            self.offset = old_shape[0]
            self.database.resize_dataset(resize_structure)  # add a new dataset to the database_path
            data_structure = {path: {'indices': np.s_[:],
                                     'columns': [0, 1, 2],
                                     'length': species_length}}
        else:
            species_length = len(self.experiment.species[species]['indices'])
            number_of_configurations = self.experiment.number_of_configurations
            dataset_structure = {path: (species_length,
                                        number_of_configurations,
                                        3)}
            self.database.add_dataset(dataset_structure)
            data_structure = {path: {'indices': np.s_[:],
                                     'columns': [0, 1, 2],
                                     'length': species_length}}

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
            data_set = data_set.prefetch(tf.data.experimental.AUTOTUNE)
            for index, batch in tqdm(enumerate(data_set), ncols=70,
                                     desc=f'{species}: Unwrapping Coordinates.'):
                data = self._transformation(batch)
                self._save_coordinates(data=data,
                                       data_structure=data_structure,
                                       index=index,
                                       batch_size=self.batch_size,
                                       system_tensor=False,
                                       tensor=True)

    def run_transformation(self):
        """
        Perform the transformation.
        """
        self._check_for_indices()
        self._unwrap_particles()  # run the transformation
