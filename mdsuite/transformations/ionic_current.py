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
        self.scale_function = {'linear': {'scale_factor': 2}}

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
        Calculate the translational dipole moment of the system.

        Returns
        -------

        """
        positions_keys = []
        charge_keys = []
        for item in data:
            if str.encode('Velocities') in item:
                positions_keys.append(item)
            elif str.encode('Charge') in item:
                charge_keys.append(item)

        if len(charge_keys) != len(positions_keys):
            charges = False
        else:
            charges = True
        system_current = tf.zeros(shape=(data[str.encode('data_size')], 3), dtype=tf.float64)
        if charges:
            for position, charge in zip(positions_keys, charge_keys):
                system_current += tf.reduce_sum(data[position] * data[charge], axis=0)
        else:
            for item in positions_keys:
                species_string = item.decode("utf-8")
                species = species_string.split('/')[0]
                # Build the charge tensor for assignment
                charge = self.experiment.species[species]['charge'][0]
                charge_tensor = tf.ones(shape=(data[str.encode('data_size')], 3), dtype=tf.float64) * charge
                system_current += tf.reduce_sum(data[item] * charge_tensor, axis=0)

        return system_current

    def _compute_ionic_current(self):
        """
        Loop over the batches, run calculations and update the database_path.
        Returns
        -------
        Updates the database_path.
        """

        type_spec = {}
        data_structure = self._prepare_database_entry()
        positions_path = [join_path(species, 'Velocities') for species in self.experiment.species]

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

        type_spec[str.encode('data_size')] = tf.TensorSpec(None, dtype=tf.int32)
        batch_generator, batch_generator_args = self.data_manager.batch_generator(dictionary=True, remainder=True)
        data_set = tf.data.Dataset.from_generator(batch_generator,
                                                  args=batch_generator_args,
                                                  output_signature=type_spec)
        data_set = data_set.prefetch(tf.data.experimental.AUTOTUNE)

        idx_start = 0
        for x in tqdm(data_set, ncols=70, desc="Ionic Current", total=self.n_batches):
            current_batch_size = int(x[str.encode('data_size')])
            data = self._transformation(x)
            self._save_coordinates(data, idx_start, current_batch_size, data_structure)
            idx_start += current_batch_size  # instead of using self.batch_size, we use  current_batch_size to take into account the reminders

    def run_transformation(self):
        """
        Run the ionic current transformation
        Returns
        -------

        """
        self._compute_ionic_current()  # run the transformation.
        self.experiment.memory_requirements = self.database.get_memory_information()
