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

import tensorflow as tf


def triu_mask(n, k=0, m=None):
    """Compute the triu mask"""
    if m is None:
        m = n
    bool_mat = tf.ones((n, m), dtype=tf.bool)
    # Just construct a boolean true matrix the size of one timestep
    if k == 0:
        return tf.linalg.band_part(bool_mat, 0, -1)
    return ~tf.linalg.band_part(bool_mat, tf.cast(-1, dtype=tf.int32), k - 1)


def triu_indices(n, k=0, m=None):
    """Replicate of np.triu_mask"""
    return tf.where(triu_mask(n, k, m))
