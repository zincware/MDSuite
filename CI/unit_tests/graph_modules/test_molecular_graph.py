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
from pathlib import Path

import numpy as np
import tensorflow as tf

from mdsuite.graph_modules.molecular_graph import (
    _apply_system_cutoff,
    build_smiles_graph,
)


class MockExperiment:
    """
    Experiment class for a unit test
    """

    database_path = Path("./")


class TestMolecularGraph:
    """
    Class to test the molecular graph module.
    """

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
            [1, 7, 3, 4.8762511, 0],
        ]
        cutoff_tensor = tf.constant(cutoff_data, dtype=tf.float32)

        # Middle range cutoff
        target = tf.constant(
            [
                [0, 0, 0, 0, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0],
            ]
        )
        mask = _apply_system_cutoff(cutoff_tensor, cutoff=5)
        np.testing.assert_array_equal(np.diagonal(mask), zeros)
        np.testing.assert_array_equal(mask, target)

        # All zeros
        target = tf.zeros((5, 5))
        mask = _apply_system_cutoff(cutoff_tensor, cutoff=0)
        np.testing.assert_array_equal(np.diagonal(mask), zeros)
        np.testing.assert_array_equal(mask, target)

        # All ones
        target = tf.ones((5, 5)) - tf.eye(5)
        mask = _apply_system_cutoff(cutoff_tensor, cutoff=200.01)
        np.testing.assert_array_equal(np.diagonal(mask), zeros)
        np.testing.assert_array_equal(mask, target)

    def test_build_smiles_graph(self):
        """
        Test the build_smiles_graph method.

        Returns
        -------
        This test checks that the SMILES graphs built by the module return the correct
        molecule information for several scenarios.
        """
        emim = "CCN1C=C[N+](+C1)C"
        bmim = "CCCCN1C=C[N+](+C1)C"
        pf = "F[P-](F)(F)(F)(F)F"  # PF6
        h2o = "[H]O[H]"
        bergenin = "OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2"
        nacl = "[Na+].[Cl-]"

        data = {
            emim: {"species": {"C": 6, "N": 2, "H": 12}, "nodes": 20},
            bmim: {"species": {"C": 8, "N": 2, "H": 16}, "nodes": 26},
            pf: {"species": {"P": 1, "F": 6}, "nodes": 7},
            h2o: {"species": {"H": 2, "O": 1}, "nodes": 3},
            bergenin: {"species": {"C": 14, "H": 16, "O": 9}, "nodes": 39},
            nacl: {"species": {"Na": 1, "Cl": 1}, "nodes": 2},
        }

        for item in data:
            graph_obj, species = build_smiles_graph(item)
            assert graph_obj.number_of_nodes() == data[item]["nodes"]
            assert species == data[item]["species"]
