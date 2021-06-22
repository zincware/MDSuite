"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.
"""

import tensorflow as tf


def unit_vector(vector):
    """
    Returns the unit vector of the vector.
    """
    return vector / tf.expand_dims(tf.norm(vector, axis=-1), -1)


def angle_between(v1, v2, acos=True):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    # return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    # return tf.math.acos(tf.clip_by_value(tf.einsum("ijk, ijk -> ij", v1_u, v2_u), -1.0, 1.0))
    if acos:
        return tf.math.acos(tf.clip_by_value(tf.einsum("ij, ij -> i", v1_u, v2_u), -1.0, 1.0))
    else:
        return tf.einsum("ij, ij -> i", v1_u, v2_u)


def get_angles(r_ij_mat, indices, acos=True):
    """
    Compute the cosine angle for the given triples

    Using :math theta = acos(r_ij * r_ik / (|r_ij| * |r_ik|))

    Parameters
    ----------
    acos: bool
        Apply tf.math.acos to the output if true. default true.
    r_ij_mat: tf.Tensor
        r_ij matrix. Shape is (n_timesteps, n_atoms, n_atoms, 3)
    indices: tf.Tensor
        Indices of the triples. Shape is (triples, 4) where a single element is composed
        of (timestep, idx, jdx, kdx)

    Returns
    -------
    tf.Tensor: Tensor with the shape (triples)
    """

    r_ij = tf.gather_nd(r_ij_mat, tf.stack([indices[:, 0], indices[:, 1], indices[:, 2]], axis=1))
    r_ik = tf.gather_nd(r_ij_mat, tf.stack([indices[:, 0], indices[:, 1], indices[:, 3]], axis=1))

    return angle_between(r_ij, r_ik, acos), 1 / (abs(r_ij) * abs(r_ik))
