"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0
Copyright Contributors to the MDSuite Project.
"""
import unittest
from mdsuite.utils.meta_functions import *


class MetaFunctionTest(unittest.TestCase):
    """
    A test class for the meta functions module.
    """
    def join_path(self):
        """
        Test the join_path method.

        Returns
        -------
        assert that join_path('a', 'b') is 'a/b'
        """
        self.assertEqual(join_path('a', 'b'), 'a/b')


if __name__ == '__main__':
    unittest.main()
