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
        return tf.math.acos(
            tf.clip_by_value(tf.einsum("ij, ij -> i", v1_u, v2_u), -1.0, 1.0)
        )
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

    r_ij = tf.gather_nd(
        r_ij_mat, tf.stack([indices[:, 0], indices[:, 1], indices[:, 2]], axis=1)
    )
    r_ik = tf.gather_nd(
        r_ij_mat, tf.stack([indices[:, 0], indices[:, 1], indices[:, 3]], axis=1)
    )

    return angle_between(r_ij, r_ik, acos), tf.linalg.norm(
        r_ij, axis=-1
    ) * tf.linalg.norm(r_ik, axis=-1)


@tf.function(experimental_relax_shapes=True)
def apply_minimum_image(r_ij, box_array):
    """

    Parameters
    ----------
    r_ij: tf.Tensor
        an r_ij matrix of size (batches, atoms, 3)
    box_array: tf.Tensor
        a box array (3,)

    Returns
    -------

    """
    return r_ij - tf.math.rint(r_ij / box_array) * box_array


def get_partial_triu_indices(n_atoms: int, m_atoms: int, idx: int) -> tf.Tensor:
    """Calculate the indices of a slice of the triu values

    Parameters
    ----------
    n_atoms: total number of atoms in the system
    m_atoms: size of the slice (horizontal)
    idx: start index of slize

    Returns
    -------
    tf.Tensor

    """
    bool_mat = tf.ones((m_atoms, n_atoms), dtype=tf.bool)
    bool_vector = ~tf.linalg.band_part(bool_mat, -1, idx)  # rename!

    indices = tf.where(bool_vector)
    indices = tf.cast(indices, dtype=tf.int32)  # is this large enough?!
    indices = tf.transpose(indices)
    return indices


def apply_system_cutoff(tensor: tf.Tensor, cutoff: float) -> tf.Tensor:
    """
    Enforce a cutoff on a tensor

    Parameters
    ----------
    tensor : tf.Tensor
    cutoff : flaot
    """

    cutoff_mask = tf.cast(tf.less(tensor, cutoff), dtype=tf.bool)  # Construct the mask

    return tf.boolean_mask(tensor, cutoff_mask)
