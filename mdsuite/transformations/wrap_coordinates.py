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
from mdsuite.transformations.transformations import Transformations
from mdsuite.utils.meta_functions import join_path
from mdsuite import experiment


def shift_positions(pos: tf.Tensor, shift):
    return pos + shift[None, None, :]


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

    def _load_data(self, species, data_name: str):
        # TODO this method should probably be a member of experiment. (building the path is not the job of the caller)
        """
        Load the tensor_values to be unwrapped
        """
        path = join_path(species, data_name)
        return tf.Tensor(self.experiment.load_matrix(path=[path], select_slice=np.s_[:]))

    def _wrap_positions(self, pos: tf.Tensor) -> tf.Tensor:
        # add extra dimensions to box_l so the direction is clear
        box_l = self.box_array[None, None, :]
        return pos - tf.floor(pos / box_l) * box_l

    def run_transformation(self):
        """
        Collect the methods in the class and unwrap the coordinates
        """

        output_name = "Positions"

        for species in self.species:

            exists = self.database.check_existence(os.path.join(species,output_name))
            # Check if the tensor_values has already been unwrapped
            if exists:
                print(f"Wrapped positions exists for {species}, "
                      f"using the saved coordinates")
            else:
                unwrap_pos = self._load_data(species, "Positions")

                if self.center_box:
                    unwrap_pos = shift_positions(unwrap_pos, self.box_array / 2.)
                wrapped_pos = self._wrap_positions(unwrap_pos)
                if self.center_box:
                    wrapped_pos = shift_positions(wrapped_pos, -self.box_array / 2.)

                path = join_path(species, output_name)
                dataset_structure = {species: {output_name: tuple(np.shape(wrapped_pos))}}
                self.database.add_dataset(dataset_structure)
                data_structure = {path: {'indices': np.s_[:],
                                         'columns': [0, 1, 2],
                                         'length': len(wrapped_pos)}}
                self._save_coordinates(data=wrapped_pos,
                                       data_structure=data_structure,
                                       index=0,
                                       batch_size=np.shape(wrapped_pos)[1],
                                       system_tensor=False,
                                       tensor=True)

        # TODO save class does not exist anymore. save somewhere else?
        #self.experiment.memory_requirements = self.database.get_memory_information()
        #self.experiment.save_class()
