""" Module for reading lammps data files"""

from mdsuite.file_io.file_read import FileProcessor


class LAMMPSTrajectoryFile(FileProcessor):
    """ Child class for the lammps file reader """

    def __init__(self, obj, header_lines=0, data_file=None):
        """ Python class constructor """
        super().__init__(obj, header_lines)
        self.data_file = data_file