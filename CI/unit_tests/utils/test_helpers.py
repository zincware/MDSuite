"""
MDSuite: A zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the zincwarecode Project.

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
Module for testing the calculator helper methods.
"""

import mdsuite
from mdsuite.utils.helpers import (
    NoneType,
    change_memory_fraction,
    compute_memory_fraction,
)


def test_change_memory_fraction():
    """Test the temporary memory fraction change."""
    assert mdsuite.config.memory_fraction == 0.5
    with change_memory_fraction(desired_memory=0.00001):
        assert mdsuite.config.memory_fraction < 0.5
    assert mdsuite.config.memory_fraction == 0.5


def test_none_type_class():
    """Test the NonType class helper."""
    my_type = NoneType

    assert my_type is not None


def test_compute_memory_fraction():
    """Test the compute memory fraction helper function."""
    # Assert for real fraction.
    memory_fraction = compute_memory_fraction(desired_memory=1.0, total_memory=2.0)
    assert memory_fraction == 0.5

    # Assert for impossible fraction
    memory_fraction = compute_memory_fraction(desired_memory=2.5, total_memory=2.0)
    assert memory_fraction == 0.9
