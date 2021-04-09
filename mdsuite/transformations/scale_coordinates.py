"""
Scale a set of coordinates by a number or vector. This is most often used when a set of cordinates is printed in
terms of a fraction of the box size and you need them to be extended. For unwrapping via printed indices, see the
unwrap_via_indices transformation.
"""

"""
Unwrap a set of coordinates based on dumped indices.
"""

from mdsuite.transformations.transformations import Transformations
from mdsuite.utils.meta_functions import join_path
import sys
import tensorflow as tf
import numpy as np


class ScaleCoordinates(Transformations):
    """ Class to scale coordinates based on dumped index values """

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
        super().__init__(experiment)
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
            path = join_path(item, 'Scaled_Positions')
            truth_table.append(self.database.check_existence(path))

        if not all(truth_table):
            print("Indices were not included in the database_path generation. Please check your simulation files.")
            sys.exit(1)

    def _transformation(self, data: tf.Tensor):
        """
        Apply the transformation to a batch of tensor_values.

        Parameters
        ----------
        data

        Returns
        -------
        Scaled coordinates : tf.Tensor
                Coordinates scaled by the image number.
        """
        return tf.math.multiply(data, self.experiment.box_array)

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
        path = join_path(species, 'Positions')
        species_length = len(self.experiment.species[species]['indices'])
        number_of_configurations = self.experiment.number_of_configurations
        dataset_structure = {path: (species_length, number_of_configurations, 3)}
        self.database.add_dataset(dataset_structure)
        data_structure = {path: {'indices': np.s_[:], 'columns': [0, 1, 2], 'length': species_length}}

        return data_structure

    def _scale_coordinates(self):
        """
        Perform the unwrapping
        Returns
        -------
        Updates the database_path object.
        """
        for species in self.species:
            data_structure = self._prepare_database_entry(species)
            data_path = [join_path(species, 'Scaled_Positions')]
            self._prepare_monitors(data_path)
            batch_generator, batch_generator_args = self.data_manager.batch_generator()
            data_set = tf.data.Dataset.from_generator(batch_generator,
                                                      args=batch_generator_args,
                                                      output_signature=tf.TensorSpec(shape=(None, self.batch_size, 3),
                                                                                     dtype=tf.float64)
                                                      )
            data_set.prefetch(tf.data.experimental.AUTOTUNE)
            for index, x in enumerate(data_set):
                data = self._transformation(x)
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
        self._scale_coordinates()  # run the transformation
