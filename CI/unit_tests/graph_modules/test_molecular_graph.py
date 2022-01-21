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
Module to test the molecular graph module.
"""
from mdsuite.graph_modules.molecular_graph import MolecularGraph

import numpy as np

from pathlib import Path
import tensorflow as tf


class MockExperiment:
    """
    Experiment class for a unit test
    """
    database_path = Path('./')


class TestMolecularGraph:
    """
    Class to test the molecular graph module.
    """
    graph_class = MolecularGraph(experiment=MockExperiment())

    def test_apply_system_cutoff(self):
        """
        Test the apply_system_cutoff method.

        Returns
        -------
        Checks whether or not the cutoff has been enforced.
        """
        zeros = np.array([0, 0, 0, 0, 0])
        cutoff_data = [
            [0, 6.8, 9, 11, 0],
            [12, 0.0, 4, 3.27, 1],
            [200, 38, 0, 1, 2.11],
            [5, 5, 5.8765342, 0, 5],
            [1, 7, 3, 4.8762511, 0]
        ]
        cutoff_tensor = tf.constant(cutoff_data, dtype=tf.float32)

        # Middle range cutoff
        target = tf.constant(
            [
                [0, 0, 0, 0, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0]
            ]
        )
        mask = self.graph_class._apply_system_cutoff(cutoff_tensor, cutoff=5)
        np.testing.assert_array_equal(np.diagonal(mask), zeros)
        np.testing.assert_array_equal(mask, target)

        # All zeros
        target = tf.zeros((5, 5))
        mask = self.graph_class._apply_system_cutoff(cutoff_tensor, cutoff=0)
        np.testing.assert_array_equal(np.diagonal(mask), zeros)
        np.testing.assert_array_equal(mask, target)

        # All ones
        target = tf.ones((5, 5)) - tf.eye(5)
        mask = self.graph_class._apply_system_cutoff(cutoff_tensor, cutoff=200.01)
        np.testing.assert_array_equal(np.diagonal(mask), zeros)
        np.testing.assert_array_equal(mask, target)
