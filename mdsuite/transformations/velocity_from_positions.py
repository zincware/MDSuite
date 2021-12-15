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
import os

import numpy as np
import tensorflow as tf

from mdsuite import experiment
from mdsuite.database.simulation_data_class import mdsuite_properties
from mdsuite.transformations.transformations import Transformations
from mdsuite.utils.meta_functions import join_path


class VelocityFromPositions(Transformations):
    """
    Calculate the velocity based on the particle positions via simple forward derivative,
    i.e. v(t) = (x(t+dt)-x(t))/dt.
    The last velocity of the trajectory cannot be computed and is copied
    from the second to last.
    """

    def __init__(self, exp: experiment.Experiment, species: list = None):
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

        self.scale_function = {"linear": {"scale_factor": 1}}

        self.storage_path = self.experiment.storage_path
        self.analysis_name = self.experiment.name
        self.output_property = mdsuite_properties.velocities_from_positions
        self.dt = self.experiment.time_step

        self.species = species
        if species is None:
            self.species = list(self.experiment.species)

    def run_transformation(self):
        """
        Perform the velocity calculation.
        """

        for species in self.species:

            exists = self.database.check_existence(
                os.path.join(species, self.output_property.name)
            )
            # Check if the tensor_values has already been unwrapped
            if exists:
                print(
                    f"{self.output_property.name} exists for {species}, "
                    "using the saved coordinates"
                )
            else:
                pos = tf.convert_to_tensor(
                    self.experiment.load_matrix(
                        property_name="Unwrapped_Positions",
                        species=[species],
                        select_slice=np.s_[:],
                    )[f"{species}/Unwrapped_Positions"]
                )
                pos_plus_dt = tf.roll(pos, shift=-1, axis=1)
                vel = (pos_plus_dt - pos) / self.dt
                # discard last value, it comes from wrapping around the positions
                vel = vel[:, :-1, :]
                # instead, append the last value again
                last_values = vel[:, -1, :]
                vel = tf.concat((vel, last_values[:, None, :]), axis=1)

                path = join_path(species, self.output_property.name)
                # TODO redundant information.
                # the only thing here that cannot be deduced here is the fact that the
                # data goes to species and not to 'Observables'
                dataset_structure = {
                    species: {self.output_property.name: tuple(np.shape(vel))}
                }
                self.database.add_dataset(dataset_structure)
                data_structure = {
                    path: {
                        "indices": np.s_[:],
                        "columns": list(range(self.output_property.n_dims)),
                        "length": len(vel),
                    }
                }
                self._save_coordinates(
                    data=vel,
                    data_structure=data_structure,
                    index=0,
                    batch_size=np.shape(vel)[1],
                    system_tensor=False,
                    tensor=True,
                )
