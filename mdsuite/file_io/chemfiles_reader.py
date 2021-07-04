"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Class for the calculation of the einstein diffusion coefficients.

Summary
-------
A module used to read trajectory files in using the chemfiles library.
"""
from chemfiles import Trajectory
import os
from mdsuite.utils.meta_functions import optimize_batch_size


class TrajectoryReader:
    """
    Read a trajectory using chemfiles.

    Attributes
    ----------
    filename : str
                Name of the file to read.
    file_format : str
            Format of the file if this is not evidenced by the file ending.
    reader : Trajectory
            Trajectory reader class with which to read the data.
    """

    def __init__(self, filename: str, file_format: str = None):
        """
        Constructor for the trajectory reader class.

        Parameters
        ----------
        filename : str
                Name of the file to read.
        file_format : str
                Format of the file if this is not evidenced by the file ending.
        """
        self.filename = filename
        self.file_format = file_format

        if file_format is not None:
            self.reader = Trajectory(filename, format=file_format)
        else:
            self.reader = Trajectory(filename)

        self.batch_size = self._compute_memory_requirements()

    def _compute_memory_requirements(self):
        """
        Compute the memory requirements of the file.

        Returns
        -------
        Updates the class state.
        """
        number_of_configurations = self.reader.nsteps
        batch_size = optimize_batch_size(self.filename,
                                         number_of_configurations)
        return batch_size

    def _get_system_information(self):
        """
        Get information about the system.

        Returns
        -------
        Either return the system information or update the database. The second
        is more modular.
        """
        pass

    def read_file(self):
        """
        Read the file.

        Returns
        -------
        Updates the simulation database.
        """


if __name__ == '__main__':
    data = Trajectory('test_data/dump.lammpstrj')
    conf_1 = data.read()

    print(data.nsteps)
