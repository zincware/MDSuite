"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.
"""

"""
Python module to calculate the integrated heat current in a experiment.
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from mdsuite.transformations.transformations import Transformations
from mdsuite.utils.meta_functions import join_path


class IntegratedHeatCurrent(Transformations):
    """
    Class to generate and store the ionic current of a experiment

    Attributes
    ----------
    experiment : object
            Experiment this transformation is attached to.
    scale_function : dict
            A dictionary referencing the memory/time scaling function of the transformation.
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
        self.scale_function = {'linear': {'scale_factor': 3}}

    def _transformation(self, data: dict) -> tf.Tensor:
        """
        Calculate the integrated thermal current of the system.

        Returns
        -------
        Integrated heat current : tf.Tensor
                Tensor values of the integrated heat current.
        """
        system_current = np.zeros((self.batch_size, 3))
        for species in self.experiment.species:
            positions_path = str.encode(join_path(species, 'Unwrapped_Positions'))
            ke_path = str.encode(join_path(species, 'KE'))
            pe_path = str.encode(join_path(species, 'PE'))
            system_current += tf.reduce_sum(data[positions_path]*(data[ke_path] + data[pe_path]), axis=0)

        return tf.convert_to_tensor(system_current)

    def _prepare_database_entry(self) -> dict:
        """
        Add the relevant tensor_values sets and groups in the database_path

        Returns
        -------
        data_structure : dict
                tensor_values structure for use in saving the tensor_values to the database_path.
        """

        number_of_configurations = self.experiment.number_of_configurations
        path = join_path('Integrated_Heat_Current', 'Integrated_Heat_Current')  # name of the new database_path
        dataset_structure = {path: (number_of_configurations, 3)}
        self.database.add_dataset(dataset_structure)  # add a new dataset to the database_path
        data_structure = {path: {'indices': np.s_[:], 'columns': [0, 1, 2]}}

        return data_structure

    def _compute_thermal_conductivity(self):
        """
        Loop over batches and compute the dipole moment
        Returns
        -------
        Performs the analysis and updates the database.
        """
        data_structure = self._prepare_database_entry()
        type_spec = {}

        species_path = [join_path(species, 'Unwrapped_Positions') for species in self.experiment.species]
        ke_path = [join_path(species, 'KE') for species in self.experiment.species]
        pe_path = [join_path(species, 'PE') for species in self.experiment.species]
        data_path = list(np.concatenate((species_path, ke_path, pe_path)))

        self._prepare_monitors(data_path)
        type_spec = self._update_species_type_dict(type_spec, species_path, 3)
        type_spec = self._update_species_type_dict(type_spec, ke_path, 1)
        type_spec = self._update_species_type_dict(type_spec, pe_path, 1)
        type_spec[str.encode('data_size')] = tf.TensorSpec(None, dtype=tf.int16)

        batch_generator, batch_generator_args = self.data_manager.batch_generator(dictionary=True, remainder=True)
        data_set = tf.data.Dataset.from_generator(batch_generator,
                                                  args=batch_generator_args,
                                                  output_signature=type_spec)
        data_set = data_set.prefetch(tf.data.experimental.AUTOTUNE)

        for idx, x in tqdm(enumerate(data_set), ncols=70, desc="Integrated Heat Current", total=self.n_batches):
            current_batch_size = int(x[str.encode('data_size')])
            data = self._transformation(x)
            self._save_coordinates(data, idx*self.batch_size, current_batch_size, data_structure)

    def run_transformation(self):
        """
        Run the ionic current transformation
        Returns
        -------
        Nothing.
        """
        self._compute_thermal_conductivity()  # run the transformation.
        self.experiment.memory_requirements = self.database.get_memory_information()
