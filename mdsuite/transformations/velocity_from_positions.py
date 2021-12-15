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
import tensorflow as tf

from mdsuite import experiment
from mdsuite.database.simulation_data_class import mdsuite_properties
from mdsuite.transformations.transformations import Transformations


class VelocityFromPositions(Transformations):
    """
    Calculate the velocity based on the particle positions via simple forward derivative,
    i.e. v(t) = (x(t+dt)-x(t))/dt.
    The last velocity of the trajectory cannot be computed and is copied
    from the second to last.
    """

    def __init__(self, exp: experiment.Experiment):
        """
        Standard constructor

        Parameters
        ----------
        exp : mdsuite.experiment.Experiment
            The experiment on which to perform the computation.
        """
        super().__init__(
            exp,
            input_properties=mdsuite_properties.unwrapped_positions,
            output_property=mdsuite_properties.velocities_from_positions,
            scale_function={"linear": {"scale_factor": 1}},
        )

        # ideally, this would also go into some input_properties entry
        self.dt = self.experiment.time_step

    def transform_batch(self, batch: tf.Tensor) -> tf.Tensor:
        pos_plus_dt = tf.roll(batch, shift=-1, axis=1)
        vel = (pos_plus_dt - batch) / self.dt
        # discard last value, it comes from wrapping around the positions
        vel = vel[:, :-1, :]
        # instead, append the last value again
        last_values = vel[:, -1, :]
        return tf.concat((vel, last_values[:, None, :]), axis=1)
