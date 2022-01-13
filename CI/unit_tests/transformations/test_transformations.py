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
In this file, we test all transformation operations
"""

import numpy as np
import tensorflow as tf

from mdsuite.database.simulation_data_class import mdsuite_properties
from mdsuite.transformations import (
    integrated_heat_current,
    ionic_current,
    kinaci_integrated_heat_current,
    momentum_flux,
    scale_coordinates,
    thermal_flux,
)
from mdsuite.transformations import translational_dipole_moment as tdp
from mdsuite.transformations import (
    unwrap_coordinates,
    unwrap_via_indices,
    velocity_from_positions,
    wrap_coordinates,
)
from mdsuite.utils.testing import assertDeepAlmostEqual

dtype = tf.float64


def test_integrated_heat_current():
    trafo = integrated_heat_current.IntegratedHeatCurrent()
    assert trafo.output_property == mdsuite_properties.integrated_heat_current
    # TODO


def test_ionic_current():
    trafo = ionic_current.IonicCurrent()
    assert trafo.output_property == mdsuite_properties.ionic_current

    n_part = 5
    n_step = 7

    input = {}
    output_should_be = np.zeros((n_step, 3))
    for sp_name in ["Na", "Cl"]:
        vel = tf.convert_to_tensor(np.random.random((n_part, n_step, 3)), dtype=dtype)
        charge = tf.convert_to_tensor([[[np.random.random()]]], dtype=dtype)

        input[sp_name] = {
            mdsuite_properties.velocities.name: vel,
            mdsuite_properties.charge.name: charge,
        }

        output_should_be += np.sum(vel.numpy() * charge.numpy(), axis=0)
    output = trafo.transform_batch(input)
    assertDeepAlmostEqual(output, output_should_be)


def test_kinaci_integrated_heat_current():
    trafo = kinaci_integrated_heat_current.KinaciIntegratedHeatCurrent()
    assert trafo.output_property == mdsuite_properties.kinaci_heat_current
    # TODO


# todo map_molecules


def test_momentum_flux():
    trafo = momentum_flux.MomentumFlux()
    assert trafo.output_property == mdsuite_properties.momentum_flux
    # TODO


def test_scale_coordinates():
    trafo = scale_coordinates.ScaleCoordinates()
    assert trafo.output_property == mdsuite_properties.scaled_positions

    n_part = 5
    n_step = 7

    pos = tf.convert_to_tensor(np.random.random((n_part, n_step, 3)), dtype=dtype)
    box_l = tf.convert_to_tensor([1.1, 2.2, 3.3], dtype=dtype)[None, None, :]

    input = {
        mdsuite_properties.positions.name: pos,
        mdsuite_properties.box_length.name: box_l,
    }
    output_should_be = pos * box_l
    output = trafo.transform_batch(input)

    assertDeepAlmostEqual(output, output_should_be)


def test_thermal_flux():
    trafo = thermal_flux.ThermalFlux()
    assert trafo.output_property == mdsuite_properties.thermal_flux
    # TODO


def test_translational_dipole_moment():
    trafo = tdp.TranslationalDipoleMoment()
    assert trafo.output_property == mdsuite_properties.translational_dipole_moment

    n_part = 5
    n_step = 7

    input = {}
    output_should_be = np.zeros((n_step, 3))
    for sp_name in ["Na", "Cl"]:
        pos = tf.convert_to_tensor(np.random.random((n_part, n_step, 3)), dtype=dtype)
        charge = tf.convert_to_tensor([[[np.random.random()]]], dtype=dtype)

        input[sp_name] = {
            mdsuite_properties.unwrapped_positions.name: pos,
            mdsuite_properties.charge.name: charge,
        }

        output_should_be += np.sum(pos.numpy() * charge.numpy(), axis=0)
    output = trafo.transform_batch(input)
    assertDeepAlmostEqual(output, output_should_be)


def test_unwrap_coordinates():
    trafo = unwrap_coordinates.CoordinateUnwrapper()
    assert trafo.output_property == mdsuite_properties.unwrapped_positions

    box_l = tf.convert_to_tensor([1.1, 2.2, 3.3], dtype=dtype)[None, None, :]
    # 1 particle, 4 time steps
    # x stays in box 0
    # y jumps 0 -> -1 -> -1 -> 0
    # z jumps 0 -> 1 -> 1 -> 2
    pos = np.array(
        [[[0.5, 0.1, 3.2]], [[0.6, 2.1, 0.9]], [[0.6, 2.1, 2.1]], [[0.6, 0.1, 0.1]]]
    )
    pos = np.swapaxes(pos, 0, 1)
    print(np.shape(pos))

    input = {
        mdsuite_properties.positions.name: tf.convert_to_tensor(pos, dtype=dtype),
        mdsuite_properties.box_length.name: box_l,
    }

    # previous carryover (same pos, but already image jumps in the last batch)
    last_carryover = {
        "last_pos": tf.convert_to_tensor([[0.5, 0.1, 3.2]], dtype=dtype),
        "last_image_box": tf.convert_to_tensor([[4, 0, 0]], dtype=dtype),
    }

    output, carryover = trafo.transform_batch(input, carryover=last_carryover)

    output_should_be = np.array(
        [
            [[4 * 1.1 + 0.5, 0.1, 3.2]],
            [[4 * 1.1 + 0.6, -0.1, 4.2]],
            [[4 * 1.1 + 0.6, -0.1, 5.4]],
            [[4 * 1.1 + 0.6, 0.1, 6.7]],
        ]
    )
    output_should_be = np.swapaxes(output_should_be, 0, 1)
    carryover_should_be = {"last_pos": [[0.6, 0.1, 0.1]], "last_image_box": [[4, 0, 2]]}
    assertDeepAlmostEqual(output.numpy(), output_should_be)
    assertDeepAlmostEqual(carryover["last_pos"], carryover_should_be["last_pos"])
    assertDeepAlmostEqual(
        carryover["last_image_box"], carryover_should_be["last_image_box"]
    )


def test_unwrap_via_indices():
    trafo = unwrap_via_indices.UnwrapViaIndices()
    assert trafo.output_property == mdsuite_properties.unwrapped_positions

    n_part = 5
    n_step = 7

    pos = tf.convert_to_tensor(np.random.random((n_part, n_step, 3)), dtype=dtype)
    box_im = tf.convert_to_tensor(
        np.random.randint(-10, 10, size=(n_part, n_step, 3)), dtype=dtype
    )
    box_l = tf.convert_to_tensor([1.1, 2.2, 3.3], dtype=dtype)[None, None, :]

    input = {
        mdsuite_properties.positions.name: pos,
        mdsuite_properties.box_images.name: box_im,
        mdsuite_properties.box_length.name: box_l,
    }
    output_should_be = pos + box_im * box_l
    output = trafo.transform_batch(input)

    assertDeepAlmostEqual(output, output_should_be)


def test_velocity_from_positions():
    trafo = velocity_from_positions.VelocityFromPositions()
    assert trafo.output_property == mdsuite_properties.velocities_from_positions

    n_part = 5
    n_step = 7

    pos = tf.convert_to_tensor(np.random.random((n_part, n_step, 3)), dtype=dtype)
    t_step = tf.convert_to_tensor([[[0.1]]], dtype=dtype)
    sample_rate = tf.convert_to_tensor([[[17]]], dtype=dtype)

    input = {
        mdsuite_properties.unwrapped_positions.name: pos,
        mdsuite_properties.time_step.name: t_step,
        mdsuite_properties.sample_rate.name: sample_rate,
    }
    output = trafo.transform_batch(input)

    vels = (pos[:, 1:, :] - pos[:, :-1, :]) / (t_step * sample_rate)
    last_vels = vels[:, -1, :]
    output_should_be = np.concatenate((vels, last_vels[:, None, :]), axis=1)

    assertDeepAlmostEqual(output, output_should_be)


def test_wrap_coordinates():
    trafo = wrap_coordinates.CoordinateWrapper(center_box=False)
    assert trafo.output_property == mdsuite_properties.positions

    n_part = 5
    n_step = 7

    pos = tf.convert_to_tensor(np.random.random((n_part, n_step, 3)), dtype=dtype)
    box_l = tf.convert_to_tensor([1.1, 2.2, 3.3], dtype=dtype)[None, None, :]

    input = {
        mdsuite_properties.unwrapped_positions.name: pos,
        mdsuite_properties.box_length.name: box_l,
    }
    output_should_be = pos - tf.floor(pos / box_l) * box_l
    output = trafo.transform_batch(input)

    assertDeepAlmostEqual(output, output_should_be)
    assert np.all(output > 0)
    assert np.all(output < box_l)
