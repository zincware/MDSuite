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
Test the molecule mapping transformation.
"""
import os
import tempfile
import unittest

from zinchub import DataHub

import mdsuite as mds


class TestMapMolecules(unittest.TestCase):
    """
    Test the map molecules module.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Prepare the class.

        Returns
        -------

        """
        cls.temp_dir = tempfile.TemporaryDirectory()
        os.chdir(cls.temp_dir.name)
        bmim_bf4 = DataHub(url="https://github.com/zincware/DataHub/tree/main/Bmim_BF4")
        bmim_bf4.get_file(path="./")

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Remove files after testing.
        """
        os.chdir("../")
        cls.temp_dir.cleanup()

    def test_molecule_mapping(self):
        """
        Test that the unwrapping runs from experiments.

        Notes
        -----
        This test will only check that the transformation runs and does not check any
        specific information about the unwrapping.
        """
        project = mds.Project()
        project.add_experiment("bmim_bf4", simulation_data="bmim_bf4.lammpstraj")
        project.experiments.bmim_bf4.run.UnwrapViaIndices()
        project.run.MolecularMap(
            molecules={
                "bmim": {"smiles": "CCCCN1C=C[N+](+C1)C", "amount": 50, "cutoff": 1.9},
                "bf4": {"smiles": "[B-](F)(F)(F)F", "amount": 50, "cutoff": 2.4},
            }
        )
        project.run.CoordinateWrapper(species=["bmim", "bf4"])
