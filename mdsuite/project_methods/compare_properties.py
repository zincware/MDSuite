"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.
"""

"""
Methods for the project class to use

Summary
-------
"""

class CompareProperties:
    """
    Methods used by the project class to compare physical properties of the experiments
    """

    def __init__(self):
        """
        Python constructor
        """

        self.x_data = None
        self.y_data_property = None
        self.y_data = None

        raise NotImplementedError