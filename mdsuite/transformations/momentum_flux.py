"""
MDSuite: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
"""
import typing

import tensorflow as tf

from mdsuite.database.simulation_data_class import mdsuite_properties
from mdsuite.transformations.transformations import MultiSpeciesTrafo


class MomentumFlux(MultiSpeciesTrafo):
    """
    Transformation to calculate the integrated heat current (positions * energies)

    """

    def __init__(self):
        super(MomentumFlux, self).__init__(
            input_properties=[mdsuite_properties.stress],
            output_property=mdsuite_properties.integrated_heat_current,
            batchable_axes=[1, 2],
            scale_function={"linear": {"scale_factor": 4}},
        )

    def transform_batch(
        self, batch: typing.Dict[str, typing.Dict[str, tf.Tensor]]
    ) -> tf.Tensor:
        fluxes = []
        for _, properties in batch.items():
            stress = properties[mdsuite_properties.stress.name]
            phi = tf.stack([stress[:, :, 3], stress[:, :, 4], stress[:, :, 5]], axis=2)
            fluxes.append(tf.reduce_sum(phi, axis=0))
        return tf.add_n(fluxes)
