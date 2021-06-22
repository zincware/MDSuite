"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Atomic transformation to unwrap the simulation coordinates.

Summary
-------
When performing analysis on the dynamics of a experiment, it often becomes necessary to reverse the effects of periodic
boundary conditions and track atoms across the box edges. This method uses the box-jump algorithm, wherein particle
positions jumps of more than a half of the box are counted as a crossing of the boundary, to allow the particles to
propagate on into space.
"""
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from mdsuite.transformations.transformations import Transformations
from mdsuite.utils.meta_functions import join_path


class CoordinateUnwrapper(Transformations):
    """
    Class to unwrap particle coordinates

    Attributes
    ----------
    experiment : object
            The experiment class from which tensor_values will be read and in which it will be saved.

    species : list
            species of atoms to unwrap.

    center_box : bool
            Decision whether or not to center the positions in the box before performing the unwrapping. The default
            value is set to True as this is most common in simulations.
    scale_function : dict
            A dictionary referencing the memory/time scaling function of the transformation.
    """

    def __init__(self, experiment: object, species: list = None, center_box: bool = True):
        """
        Constructor for the Ionic current calculator.

        Parameters
        ----------
        experiment : object
                Experiment this transformation is attached to.
        species : list
                Atomic species on which to perform this operation
        center_box : bool
                If true, the origin of the coordinates will be centered.
        """
        super().__init__(experiment)
        self.scale_function = {'linear': {'scale_factor': 3}}
        self.center_box = center_box

        if species is None:
            self.species = list(self.experiment.species)
        else:
            self.species = self.species

    def _prepare_database_entry(self, species: str):
        """
        Call some housekeeping methods and prepare for the transformations.
        Returns
        -------

        """
        # collect machine properties and determine batch size
        path = join_path(species, 'Unwrapped_Positions')  # name of the new database_path
        dataset_structure = {path: (len(self.experiment.species[species]['indices']),
                                    self.experiment.number_of_configurations,
                                    3)}
        self.database.add_dataset(dataset_structure)  # add a new dataset to the database_path
        data_structure = {path: {'indices': np.s_[:], 'columns': [0, 1, 2]}}

        return data_structure

    @staticmethod
    def _calculate_difference_tensor(data: tf.Tensor):
        """
        Calculate the amount particles move in each time step

        Returns
        -------
        distance tensor : tf.tensor
                Returns the tensor of distances in time.
        """

        return tf.experimental.numpy.diff(data, axis=1)

    def _construct_image_mask(self, data: tf.Tensor, current_state: tf.Tensor = None):
        """
        Build an image mask on a set of data.

        Returns
        -------
        Cumulative image mask.
        """
        distance_tensor = self._calculate_difference_tensor(data)  # get the distance tensor
        # Find all distance greater than half a box length and set them to integers.
        mask = tf.cast(tf.cast(tf.greater_equal(abs(distance_tensor), np.array(self.experiment.box_array) / 2),
                               dtype=tf.int16), dtype=tf.float64)

        mask = tf.multiply(tf.sign(distance_tensor), mask)  # get the correct image sign

        mask = tf.map_fn(lambda x: tf.concat(([[0, 0, 0]], x), axis=0), tf.math.cumsum(mask, axis=1), dtype=tf.float64)
        correction = tf.cast(tf.repeat(tf.expand_dims(current_state, 1), mask.shape[1], 1), dtype=tf.float64)
        corrected_mask = tf.add(mask, correction)

        return corrected_mask

    def _apply_mask(self, data: tf.Tensor, mask: tf.Tensor):
        """
        Apply the unwrapping mask to the data.

        Returns
        -------
        Unwrapped coordinate positions.
        """
        scaled_mask = mask * tf.convert_to_tensor(self.experiment.box_array, dtype=tf.float64)  # get the scaled mask

        return data - scaled_mask  # apply the scaling

    def _transformation(self, data: np.array, state: tf.Tensor, last_conf: tf.Tensor = None):
        """
        Perform the unwrapping transformation on the data.

        Returns
        -------

        """
        if self.center_box:
            data[:, :, 0] -= (self.experiment.box_array[0] / 2)
            data[:, :, 1] -= (self.experiment.box_array[1] / 2)
            data[:, :, 2] -= (self.experiment.box_array[2] / 2)

        data = tf.convert_to_tensor(data)
        if last_conf is not None:
            data = tf.concat((tf.expand_dims(last_conf, 1), data), axis=1)
        mask = self._construct_image_mask(data, current_state=state)
        return self._apply_mask(data, mask), mask[:, -1], data[:, -1]

    def _unwrap_coordinates(self, species: str):
        """
        Loop over the batches, run calculations and update the database_path.
        Returns
        -------
        Updates the database_path.
        """

        type_spec = {}
        data_structure = self._prepare_database_entry(species)
        positions_path = [join_path(species, 'Positions')]
        self._prepare_monitors(positions_path)
        type_spec = self._update_species_type_dict(type_spec, positions_path, 3)
        type_spec[str.encode('data_size')] = tf.TensorSpec(None, dtype=tf.int32)
        batch_generator, batch_generator_args = self.data_manager.batch_generator(dictionary=True, remainder=True)
        data_set = tf.data.Dataset.from_generator(batch_generator,
                                                  args=batch_generator_args,
                                                  output_signature=type_spec)
        data_set = data_set.prefetch(tf.data.experimental.AUTOTUNE)
        state = tf.zeros(shape=(len(self.experiment.species[species]['indices']), 3))
        last_conf = tf.zeros(shape=(len(self.experiment.species[species]['indices']), 3))
        loop_correction = self._remainder_to_binary()
        for index, x in tqdm(enumerate(data_set), ncols=70, desc=f"{species} unwrapping",
                             total=self.n_batches+loop_correction):
            if index == 0:
                data, state, last_conf = self._transformation(np.array(x[str.encode(join_path(species, 'Positions'))]),
                                                              state)
                self._save_coordinates(data,
                                       index * self.batch_size,
                                       x[str.encode('data_size')],
                                       data_structure,
                                       system_tensor=False,
                                       tensor=True)
            else:
                data, state, last_conf = self._transformation(np.array(x[str.encode(join_path(species, 'Positions'))]),
                                                              state, last_conf)
                self._save_coordinates(data[:, 1:],
                                       index*self.batch_size,
                                       x[str.encode('data_size')],
                                       data_structure,
                                       system_tensor=False,
                                       tensor=True)

    def run_transformation(self):
        """
        Run the ionic current transformation
        Returns
        -------

        """
        for item in self.species:
            self._unwrap_coordinates(item)  # run the transformation.
            self.experiment.memory_requirements = self.database.get_memory_information()
