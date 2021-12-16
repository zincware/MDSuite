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
Test coordinate unwrapping via indices.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np
from zinchub import DataHub

import mdsuite as mds
import mdsuite.transformations


class TestUnwrapViaIndices(unittest.TestCase):
    """
    A test class for the UnwrapViaIndices module.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Prepare the class.

        Returns
        -------

        """
        cls.test_dir = tempfile.mkdtemp()
        os.chdir(cls.test_dir)

        NaCl = DataHub(url="https://github.com/zincware/DataHub/tree/main/NaCl_gk_i_q")
        NaCl.get_file(path="./")

        return (temporary_path / NaCl.file_raw).as_posix()

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Move out of tmp directory and delete it at end of test.
        Returns
        -------

        """
