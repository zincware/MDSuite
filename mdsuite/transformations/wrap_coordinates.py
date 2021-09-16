"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Atomic transformation to wrap the simulation coordinates.

Summary
-------
Sometimes, particularly in the study of molecules, it is necessary to wrap
positions. This module will do that for you.
"""
import numpy as np
import os
import tensorflow as tf
import tqdm
from mdsuite.transformations.transformations import Transformations
from mdsuite.utils.meta_functions import join_path
from mdsuite import experiment


class CoordinateWrapper(Transformations):
    """
    Class to wrap particle coordinates

    Attributes
    ----------
    experiment : object
            The experiment class from which tensor_values will be read and in
            which it will be saved.

    species : list
            species of atoms to unwrap.

    center_box : bool
            Decision whether or not to center the positions in the box before
            performing the unwrapping. The default value is set to True as
            this is most common in simulations.

    storage_path : str
            Path to the tensor_values in the database_path, taken directly
            from the system attribute.

    analysis_name : str
            Name of the analysis, taken directly from the system attribute.

    box_array : list
            Box array of the simulation.

    scale_function : dict
            A dictionary referencing the memory/time scaling function
            of the transformation.
    """

    def __init__(self,
                 exp: experiment.Experiment,
                 species: list = None,
                 center_box: bool = True):
        """
        Standard constructor

        Parameters
        ----------
        exp : mdsuite.experiment.Experiment
                Experiment object to use and update.
        species : list
                List of species to perform unwrapping on
        center_box : bool
                If the box is centered around the origin. Default: True
        """
        super().__init__(exp)

        self.scale_function = {'linear': {'scale_factor': 1}}

        self.storage_path = self.experiment.storage_path
        self.analysis_name = self.experiment.name
        self.center_box = center_box

        self.box_array = np.asarray(self.experiment.box_array)
        self.species = species
        if species is None:
            self.species = list(self.experiment.species)

    def _wrap_positions(self, unwrap_pos: tf.Tensor) -> tf.Tensor:
        """
        Do the actual wrapping of positions

        Parameters
        ----------
        unwrap_pos : tf.Tensor
            The coordinates to wrap

        Returns
        -------
        tf.Tensor containing the wrapped coordinates. Same shape as unwrap_pos.
        """

        box_l = self.box_array[None, None, :]

        if self.center_box:
            unwrap_pos = unwrap_pos + box_l / 2.
        pos = unwrap_pos - tf.floor(unwrap_pos / box_l) * box_l
        if self.center_box:
            pos = pos - box_l / 2.

        return pos

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
        existing = self._run_dataset_check(path)
        if existing:
            old_shape = self.database.get_data_size(path)
            species_length = len(self.experiment.species[species]['indices'])
            resize_structure = {path: (species_length, self.experiment.number_of_configurations - old_shape[0], 3)}
            self.offset = old_shape[0]
            self.database.resize_dataset(resize_structure)  # add a new dataset to the database_path
            data_structure = {path: {'indices': np.s_[:], 'columns': [0, 1, 2], 'length': species_length}}
        else:
            species_length = len(self.experiment.species[species]['indices'])
            number_of_configurations = self.experiment.number_of_configurations
            dataset_structure = {path: (species_length, number_of_configurations, 3)}
            self.database.add_dataset(dataset_structure)
            data_structure = {path: {'indices': np.s_[:], 'columns': [0, 1, 2], 'length': species_length}}

        return data_structure

    def run_transformation(self):
        """
        Wrap coordinates batch by batch.
        """

        output_name = "Positions"

        for species in self.species:
            exists = self.database.check_existence(os.path.join(species, output_name))
            if exists:
                print(f"Wrapped positions exists for {species}, "
                      f"using the saved coordinates")
            else:

                data_structure = self._prepare_database_entry(species)
                data_path = [join_path(species, output_name)]
                self._prepare_monitors(data_path)
                batch_generator, batch_generator_args = self.data_manager.batch_generator()
                data_set = tf.data.Dataset.from_generator(batch_generator,
                                                          args=batch_generator_args,
                                                          output_signature=tf.TensorSpec(
                                                              shape=(None, self.batch_size, 3),
                                                              dtype=tf.float64)
                                                          )
                data_set = data_set.prefetch(tf.data.experimental.AUTOTUNE)
                for index, x in tqdm.tqdm(enumerate(data_set),
                                          ncols=70,
                                          desc=f'{species}: Wrapping Coordinates'):
                    unwrap_pos = tf.convert_to_tensor(self.experiment.load_matrix(identifier="Unwrapped_Positions",
                                                                                  species=species))
                    wrapped_pos = self._wrap_positions(unwrap_pos)
                    self._save_coordinates(data=wrapped_pos,
                                           data_structure=data_structure,
                                           index=index,
                                           batch_size=self.batch_size,
                                           system_tensor=False,
                                           tensor=True)
