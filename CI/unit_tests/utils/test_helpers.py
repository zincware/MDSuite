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
Test the helper functions module.
"""
from mdsuite.utils.helpers import generate_dataclass


class TestHelpers:
    """
    Test all functions in the helpers file.
    """

    def test_generate_dataclass(self):
        """
        Test the generate dataclass function.
        """
        class_1 = generate_dataclass(par_a=5, par_b="hello", par_c=None)
        assert class_1.par_a == 5
        assert class_1.par_b == "hello"
        assert class_1.par_c is None
