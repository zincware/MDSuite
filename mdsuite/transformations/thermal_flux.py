"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Python module to calculate the ionic current in a experiment.
"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from mdsuite.transformations.transformations import Transformations
from mdsuite.utils.meta_functions import join_path


class ThermalFlux(Transformations):
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
        self.scale_function = {'linear': {'scale_factor': 5}}

    def _prepare_database_entry(self):
        """
        Call some housekeeping methods and prepare for the transformations.
        Returns
        -------

        """
        path = join_path('Thermal_Flux', 'Thermal_Flux')  # name of the new database_path
        existing = self._run_dataset_check(path)
        if existing:
            old_shape = self.database.get_data_size(path)
            resize_structure = {path: (self.experiment.number_of_configurations - old_shape[0], 3)}
            self.offset = old_shape[0]
            self.database.resize_dataset(resize_structure)  # add a new dataset to the database_path
            data_structure = {path: {'indices': np.s_[:, ], 'columns': [0, 1, 2]}}
        else:
            dataset_structure = {path: (self.experiment.number_of_configurations,
                                        3)}
            self.database.add_dataset(dataset_structure)  # add a new dataset to the database_path
            data_structure = {path: {'indices': np.s_[:],
                                     'columns': [0, 1, 2]}}

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
            velocity_path = str.encode(join_path(species, 'Velocities'))
            ke_path = str.encode(join_path(species, 'KE'))
            pe_path = str.encode(join_path(species, 'PE'))
            phi_x = np.multiply(data[stress_path][:, :, 0], data[velocity_path][:, :, 0]) + \
                np.multiply(data[stress_path][:, :, 3], data[velocity_path][:, :, 1]) + \
                np.multiply(data[stress_path][:, :, 4], data[velocity_path][:, :, 2])
            phi_y = np.multiply(data[stress_path][:, :, 3], data[velocity_path][:, :, 0]) + \
                np.multiply(data[stress_path][:, :, 1], data[velocity_path][:, :, 1]) + \
                np.multiply(data[stress_path][:, :, 5], data[velocity_path][:, :, 2])
            phi_z = np.multiply(data[stress_path][:, :, 4], data[velocity_path][:, :, 0]) + \
                np.multiply(data[stress_path][:, :, 5], data[velocity_path][:, :, 1]) + \
                np.multiply(data[stress_path][:, :, 2], data[velocity_path][:, :, 2])

            phi = np.dstack([phi_x, phi_y, phi_z])

            phi_sum = phi.sum(axis=0)
            phi_sum_atoms = phi_sum / self.experiment.units['NkTV2p']  # factor for units lammps nktv2p

            energy = data[ke_path] + data[pe_path]

            energy_velocity = energy * data[velocity_path]
            energy_velocity_atoms = tf.reduce_sum(energy_velocity, axis=0)
            system_current += energy_velocity_atoms - phi_sum_atoms

        return system_current

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
            dictionary[str.encode(item)] = tf.TensorSpec(shape=(None,
                                                                self.batch_size,
                                                                dimension),
                                                         dtype=tf.float64)

        return dictionary

    def _compute_thermal_flux(self):
        """
        Loop over the batches, run calculations and update the database_path.
        Returns
        -------
        Updates the database_path.
        """

        data_structure = self._prepare_database_entry()
        type_spec = {}

        species_path = [join_path(species, 'Velocities') for species in self.experiment.species]
        stress_path = [join_path(species, 'Stress') for species in self.experiment.species]
        pe_path = [join_path(species, 'PE') for species in self.experiment.species]
        ke_path = [join_path(species, 'KE') for species in self.experiment.species]

        data_path = np.concatenate((species_path, stress_path, pe_path, ke_path))
        self._prepare_monitors(data_path)
        type_spec = self._update_species_type_dict(type_spec, species_path, 3)
        type_spec = self._update_species_type_dict(type_spec, stress_path, 6)
        type_spec = self._update_species_type_dict(type_spec, pe_path, 1)
        type_spec = self._update_species_type_dict(type_spec, ke_path, 1)

        type_spec[str.encode('data_size')] = tf.TensorSpec(None, dtype=tf.int16)
        batch_generator, batch_generator_args = self.data_manager.batch_generator(dictionary=True, remainder=True)
        data_set = tf.data.Dataset.from_generator(batch_generator,
                                                  args=batch_generator_args,
                                                  output_signature=type_spec)
        data_set = data_set.prefetch(tf.data.experimental.AUTOTUNE)

        for idx, x in tqdm(enumerate(data_set), ncols=70, desc="Thermal Flux", total=self.n_batches):
            current_batch_size = int(x[str.encode('data_size')])
            data = self._transformation(x)
            self._save_coordinates(data, idx * self.batch_size,
                                   current_batch_size,
                                   data_structure)

    def run_transformation(self):
        """
        Run the ionic current transformation
        Returns
        -------

        """
        self._compute_thermal_flux()  # run the transformation.
        self.experiment.memory_requirements = self.database.get_memory_information()
