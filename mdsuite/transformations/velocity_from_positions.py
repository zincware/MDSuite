"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Summary
-------
Calculate the velocity of particles from their positions
"""
import numpy as np
import os
import tensorflow as tf
from mdsuite.transformations.transformations import Transformations
from mdsuite.utils.meta_functions import join_path
from mdsuite import experiment


class VelocityFromPositions(Transformations):
    """
    Calculate the velocity based on the particle positions via simple forward derivative, i.e.
    v(t) = (x(t+dt)-x(t))/dt. The last velocity of the trajectory cannot be computed and is copied
    from the second to last.
    """

    def __init__(self,
                 exp: experiment.Experiment,
                 species: list = None):
        """
        Standard constructor

        Parameters
        ----------
        exp : mdsuite.experiment.Experiment
            The experiment on which to perform the computation.
        species : list, optional
            A list of species for which to perform the computation. Default: All species.
        """
        super().__init__(exp)

        self.scale_function = {'linear': {'scale_factor': 1}}

        self.storage_path = self.experiment.storage_path
        self.analysis_name = self.experiment.name
        self.dt = self.experiment.time_step

        self.species = species
        if species is None:
            self.species = list(self.experiment.species)

    def run_transformation(self):
        """
        Perform the velocity calculation.
        """
        output_name = "Velocities_From_Positions"
        for species in self.species:

            exists = self.database.check_existence(os.path.join(species, output_name))
            # Check if the tensor_values has already been unwrapped
            if exists:
                print(f"{output_name} exists for {species}, "
                      f"using the saved coordinates")
            else:
                pos = tf.convert_to_tensor(self.experiment.load_matrix(identifier="Unwrapped_Positions",
                                                                       species=species,
                                                                       select_slice=np.s_[:]))
                pos_plus_dt = tf.roll(pos, axis=1, shift=-1)
                vel = (pos_plus_dt - pos) / self.dt
                # discard last value, it comes from wrapping around the positions
                vel = vel[:, -1:, :]
                # instead, append the last value again
                vel = tf.concat(vel, vel[:, -1, :], axis=1)

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
