"""
Python module to calculate the ionic current in a experiment.
"""

import numpy as np
from tqdm import tqdm
import tensorflow as tf

from mdsuite.transformations.transformations import Transformations
from mdsuite.utils.meta_functions import join_path


class IonicCurrent(Transformations):
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
        super().__init__(experiment)

    def _prepare_database_entry(self):
        """
        Call some housekeeping methods and prepare for the transformations.
        Returns
        -------

        """
        # collect machine properties and determine batch size
        path = join_path('Ionic_Current', 'Ionic_Current')  # name of the new database_path
        dataset_structure = {path: (self.experiment.number_of_configurations, 3)}
        self.database.add_dataset(dataset_structure)  # add a new dataset to the database_path
        data_structure = {path: {'indices': np.s_[:], 'columns': [0, 1, 2]}}

        return data_structure

    def _transformation(self, data: tf.Tensor):
        """
        Compute the ionic current of the experiment.

        Parameters
        ----------
        batch_number
        remainder
        Returns
        -------
        system_current : np.array
                System current as a numpy array.
        """

        system_current = tf.reduce_sum(data, axis=1)

        # build charge array
        system_charges = [self.experiment.species[atom]['charge'][0] for atom in self.experiment.species]

        # Calculate the total experiment current
        charge_tuple = []  # define empty array for the charges
        for charge in system_charges:  # loop over each species charge
            # Build a tensor of charges allowing for memory management.
            charge_tuple.append(tf.ones([self.batch_size, 3], dtype=tf.float64) * charge)

        charge_tensor = tf.stack(charge_tuple)  # stack the tensors into a single object
        system_current = tf.reduce_sum(system_current*charge_tensor, axis=0)

        return system_current

    def _compute_ionic_current(self):
        """
        Loop over the batches, run calculations and update the database_path.
        Returns
        -------
        Updates the database_path.
        """

        data_structure = self._prepare_database_entry()
        data_path = [join_path(species, 'Velocities') for species in self.experiment.species]
        self._prepare_monitors(data_path)
        batch_generator, batch_generator_args = self.data_manager.batch_generator()
        data_set = tf.data.Dataset.from_generator(batch_generator,
                                                  args=batch_generator_args,
                                                  output_signature=tf.TensorSpec(shape=(len(data_path),
                                                                                        None,
                                                                                        None,
                                                                                        3), dtype=tf.float64))
        data_set = data_set.prefetch(tf.data.experimental.AUTOTUNE)
        for index, x in enumerate(data_set):
            data = self._transformation(x)
            self._save_coordinates(data, index, self.batch_size, data_structure)

    def run_transformation(self):
        """
        Run the ionic current transformation
        Returns
        -------

        """
        self._compute_ionic_current()  # run the transformation.
        self.experiment.memory_requirements = self.database.get_memory_information()
