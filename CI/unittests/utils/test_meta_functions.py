"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0
Copyright Contributors to the MDSuite Project.
"""
import unittest
from mdsuite.utils.meta_functions import *
import os


class TestMetaFunction(unittest.TestCase):
    """
    A test class for the meta functions module.
    """
    def test_join_path(self):
        """
        Test the join_path method.

        Returns
        -------
        assert that join_path('a', 'b') is 'a/b'
        """
        self.assertEqual(join_path('a', 'b'), 'a/b')

    def test_get_dimensionality(self):
        """
        Test the get_dimensionality method.

        Returns
        -------
        assert that for all choices of dimension array the correct dimension comes out.
        """
        one_d = [1, 0, 0]
        two_d = [1, 1, 0]
        three_d = [1, 1, 1]
        self.assertEqual(get_dimensionality(one_d), 1)
        self.assertEqual(get_dimensionality(two_d), 2)
        self.assertEqual(get_dimensionality(three_d), 3)

    def test_get_machine_properties(self):
        """
        Test the get_machine_properties method.

        Returns
        -------
        This test will just run the method and check for a failure.
        """
        get_machine_properties()

    def test_line_counter(self):
        """
        Test the line_counter method.

        Returns
        -------
        Check that the correct number of lines is return for a test file.
        """
        data = [["a\n"], ["b\n"], ["c\n"], ["d"]]
        name = 'line_counter_test.txt'
        with open(name, 'w') as f:
            for item in data:
                f.write(item[0])
        self.assertEqual(line_counter(name), 4)
        os.remove(name)

    def test_optimize_batch_size(self):
        """
        Test the optimize_batch_size method.

        Returns
        -------
        assert the correct batch size is returned for several inputs.
        """
        # Assert that the batch number is the full trajectory.
        number_of_configurations = 10
        _file_size = 100
        _memory = 1000000
        self.assertEqual(optimize_batch_size('None', number_of_configurations, _file_size, _memory, test=True), 10)

        # Assert that the batch number is the half the trajectory.
        number_of_configurations = 10
        _file_size = 100
        _memory = 250
        self.assertEqual(optimize_batch_size('None', number_of_configurations, _file_size, _memory, test=True), 5)


if __name__ == '__main__':
    unittest.main()
