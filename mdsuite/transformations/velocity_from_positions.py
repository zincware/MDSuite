"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Atomic transformation to wrap the simulation coordinates.

Summary
-------
Calculate the velocity of particles from their positions via simple v = delta x/ delta t
"""
import numpy as np
import os
import tensorflow as tf
from mdsuite.transformations.transformations import Transformations
from mdsuite.utils.meta_functions import join_path
from mdsuite import experiment


class VelocityFromPositions(Transformations):

    def __init__(self,
                 exp: experiment.Experiment,
                 species: list = None):
        super().__init__(exp)

        self.scale_function = {'linear': {'scale_factor': 1}}

        self.storage_path = self.experiment.storage_path
        self.analysis_name = self.experiment.name
        self.dt = self.experiment.time_step

        self.species = species
        if species is None:
            self.species = list(self.experiment.species)

    def _load_data(self, species):
        path = join_path(species, "Unwrapped_Positions")
        return tf.convert_to_tensor(self.experiment.load_matrix(path=[path], select_slice=np.s_[:]))

    def run_transformation(self):
        output_name = "Velocities_From_Positions"
        for species in self.species:

            exists = self.database.check_existence(os.path.join(species, output_name))
            # Check if the tensor_values has already been unwrapped
            if exists:
                print(f"Velocities from Positions exists for {species}, "
                      f"using the saved coordinates")
            else:
                pos = self._load_data(species)
                pos_rolled = tf.roll(pos, axis=1, shift=1)
                vel = (pos- pos_rolled)/ self.dt
                # discard first value, it comes from wrapping around the positions
                vel = vel[:, 1:, :]

                path = join_path(species, output_name)
                dataset_structure = {species: {output_name: tuple(np.shape(vel))}}
                self.database.add_dataset(dataset_structure)
                data_structure = {path: {'indices': np.s_[:],
                                         'columns': [0, 1, 2],
                                         'length': len(vel)}}
                self._save_coordinates(data=vel,
                                       data_structure=data_structure,
                                       index=0,
                                       batch_size=np.shape(vel)[1],
                                       system_tensor=False,
                                       tensor=True)
