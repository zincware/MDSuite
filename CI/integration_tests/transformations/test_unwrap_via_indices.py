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
Test unwrapping via indices.
"""
import unittest
import os

from zinchub import DataHub

import tempfile

import mdsuite as mds


class TestUnwrapViaIndices(unittest.TestCase):
    """
    Test the memory manager module.
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
        NaCl = DataHub(url="https://github.com/zincware/DataHub/tree/main/NaCl_gk_i_q")
        NaCl.get_file(path='./')

        cls.project = mds.Project()
        cls.project.add_experiment("NaCl", simulation_data='NaCl_gk_i_q.lammpstraj')

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Remove files after testing.
        """
        os.chdir('../')
        cls.temp_dir.cleanup()

    def test_from_project(self):
        """
        Test calling the transformation from the project class.

        Notes
        -----
        This test will only check that the transformation runs and does not check any
        specific information about the unwrapping.
        """
        self.project.run.UnwrapViaIndices()

    def test_from_experiment(self):
        """
        Test that the unwrapping runs from experiments.

        Notes
        -----
        This test will only check that the transformation runs and does not check any
        specific information about the unwrapping.
        """
        self.project.experiments.NaCl.run.UnwrapViaIndices()

    def test_new_data_unwrapping(self):
        """
        Test how the transformation works if new data is added.

        Notes
        -----
        Checks if the inital unwrapping works, adds new data, and checks that the
        unwrapping occurs again. No explicit test for if the datasets in the hdf5
        database have been correctly resized.
        """
        self.project.run.UnwrapViaIndices()
        self.project.experiments.NaCl.add_data('NaCl_gk_i_q.lammpstraj', force=True)
        self.project.run.UnwrapViaIndices()
