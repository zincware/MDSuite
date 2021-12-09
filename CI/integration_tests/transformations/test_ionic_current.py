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
Test the ionic current transformation.
"""
import os
import tempfile
import unittest

from zinchub import DataHub

import mdsuite as mds


class TestIonicCurrent(unittest.TestCase):
    """
    Test the ionic current module.
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
        NaCl.get_file(path="./")

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Remove files after testing.
        """
        os.chdir("../")
        cls.temp_dir.cleanup()

    def test_from_project(self):
        """
        Test calling the transformation from the project class.

        Notes
        -----
        This test will only check that the transformation runs and does not check any
        specific information about the unwrapping.
        """
        project = mds.Project(name="my_poject_5")
        project.add_experiment("NaCl", simulation_data="NaCl_gk_i_q.lammpstraj")
        project.run.IonicCurrent()

    def test_from_experiment(self):
        """
        Test that the transformation runs from experiments.

        Notes
        -----
        This test will only check that the transformation runs and does not check any
        specific information about the unwrapping.
        """
        project = mds.Project()
        project.add_experiment("NaCl1", simulation_data="NaCl_gk_i_q.lammpstraj")
        project.experiments.NaCl1.run.IonicCurrent()

    def test_new_data_transformation(self):
        """
        Test how the transformation works if new data is added.

        Notes
        -----
        Checks if the initial transformation works, adds new data, and checks that the
        transformation occurs again. No explicit test for if the datasets in the hdf5
        database have been correctly resized.
        """
        project = mds.Project()
        project.add_experiment("NaCl2", simulation_data="NaCl_gk_i_q.lammpstraj")
        project.experiments.NaCl2.run.IonicCurrent()
        project.experiments.NaCl2.add_data("NaCl_gk_i_q.lammpstraj", force=True)
        project.experiments.NaCl2.run.IonicCurrent()
