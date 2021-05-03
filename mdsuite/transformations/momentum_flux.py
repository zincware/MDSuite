"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.
"""

"""
Python module to calculate the momentum flux in an experiment.
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from mdsuite.transformations.transformations import Transformations
from mdsuite.utils.meta_functions import join_path


class MomentumFlux(Transformations):
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
        self.scale_function = {'linear': {'scale_factor': 4}}

    def _prepare_database_entry(self):
        """
        Call some housekeeping methods and prepare for the transformations.
        Returns
        -------

        """
        # collect machine properties and determine batch size
        path = join_path('Momentum_Flux', 'Momentum_Flux')  # name of the new database_path
        existing = self._run_dataset_check(path)
        if existing:
            old_shape = self.database.get_data_size(path)
            resize_structure = {path: (self.experiment.number_of_configurations - old_shape[0], 3)}
            self.offset = old_shape[0]
            self.database.resize_dataset(resize_structure)  # add a new dataset to the database_path
            data_structure = {path: {'indices': np.s_[:, ], 'columns': [0, 1, 2]}}
        else:
            dataset_structure = {path: (self.experiment.number_of_configurations, 3)}
            self.database.add_dataset(dataset_structure)  # add a new dataset to the database_path
            data_structure = {path: {'indices': np.s_[:], 'columns': [0, 1, 2]}}

        return data_structure

    def _transformation(self, data: tf.Tensor):
        """
        Compute the ionic current of the experiment.

        Parameters
        ----------
        data : tf.Tensor
                Data on which to apply the operation.
        Returns
        -------
        system_current : np.array
                System current as a numpy array.
        """

        system_current = np.zeros((self.batch_size, 3))
        for species in self.experiment.species:
            stress_path = str.encode(join_path(species, 'Stress'))
            phi_x = data[stress_path][:, :, 3]
            phi_y = data[stress_path][:, :, 4]
            phi_z = data[stress_path][:, :, 5]

            phi = np.dstack([phi_x, phi_y, phi_z])

            system_current += tf.reduce_sum(phi, axis=0)

        return system_current

    def _compute_momentum_flux(self):
        """
        Loop over the batches, run calculations and update the database_path.
        Returns
        -------
        Updates the database_path.
        """

        data_structure = self._prepare_database_entry()
        type_spec = {}
        data_path = [join_path(species, 'Stress') for species in self.experiment.species]
        self._prepare_monitors(data_path)

        type_spec = self._update_species_type_dict(type_spec, data_path, 6)
        type_spec[str.encode('data_size')] = tf.TensorSpec(None, dtype=tf.int16)
        batch_generator, batch_generator_args = self.data_manager.batch_generator(dictionary=True, remainder=True)
        data_set = tf.data.Dataset.from_generator(batch_generator,
                                                  args=batch_generator_args,
                                                  output_signature=type_spec)
        data_set = data_set.prefetch(tf.data.experimental.AUTOTUNE)

        for idx, x in tqdm(enumerate(data_set), ncols=70, desc="Momentum Flux", total=self.n_batches):
            current_batch_size = int(x[str.encode('data_size')])
            data = self._transformation(x)
            self._save_coordinates(data, idx*self.batch_size, current_batch_size, data_structure)


def run_transformation(self):
        """
        Run the ionic current transformation
        Returns
        -------

        """
        self._compute_momentum_flux()  # run the transformation.
        self.experiment.memory_requirements = self.database.get_memory_information()
