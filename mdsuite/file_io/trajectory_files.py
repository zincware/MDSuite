"""
Parent class for file processing

Summary
-------
"""

import abc
from typing import TextIO
import numpy as np
from mdsuite.utils.meta_functions import join_path
from mdsuite.file_io.file_read import FileProcessor


class TrajectoryFile(FileProcessor, metaclass=abc.ABCMeta):
    """
    Parent class for file reading and processing

    Attributes
    ----------
    obj, project : object
            File object to be opened and read in.
    header_lines : int
            Number of header lines in the file format being read.
    """

    def __init__(self, obj, header_lines, file_path, sort: bool = False):
        """
        Python constructor

        Parameters
        ----------
        obj : object
                Experiment class instance to add to.

        header_lines : int
                Number of header lines in the given file format.

        sort : bool
                If true, the tensor_values in the trajectory file must be sorted during the database_path build.
        """

        super().__init__(obj, header_lines, file_path)  # fill the experiment class
        self.sort = sort

    def _read_header(self, f: TextIO, offset: int = 0) -> list:
        """
        Read n header lines in starting from line offset.

        Parameters
        ----------
        f : TextIO
                File object to read from
        offset : int
                Number of lines to skip before reading in the header
        Returns
        -------
        header : list
                list of tensor_values in the header
        """

        # Skip the offset tensor_values
        for i in range(offset):
            f.readline()

        return [next(f).split() for _ in range(self.header_lines)]  # Get the first header

    def read_configurations(self, number_of_configurations: int, file_object: TextIO, line_length: int):
        """
        Read in a number of configurations from a file

        Parameters
        ----------
        line_length : int
                Length of each line of tensor_values to be read in. Necessary for instantiation.
        number_of_configurations : int
                Number of configurations to be read in.
        file_object : experiment
                File object to be read from.

        Returns
        -------
        configuration tensor_values : np.array
                Data read in from the file object.
        """

        # Define the empty tensor_values array
        configurations_data = np.empty((number_of_configurations * self.experiment.number_of_atoms, line_length),
                                       dtype='<U15')

        counter = 0
        for i in range(number_of_configurations):

            for j in range(self.header_lines):
                file_object.readline()

            # Read the tensor_values into the arrays.
            for k in range(self.experiment.number_of_atoms):
                configurations_data[counter] = np.array(list(file_object.readline().split()))
                counter += 1  # update the counter
        return configurations_data

    def build_file_structure(self, batch_size: int = None):
        """
        Build a skeleton of the file so that the database_path class can process it correctly.
        """

        structure = {}  # define initial dictionary
        if batch_size is None:
            batch_size = self.experiment.batch_size

        for item in self.experiment.species:
            positions = np.array([np.array(self.experiment.species[item]['indices']) + i * self.experiment.number_of_atoms -
                                  self.header_lines for i in range(batch_size)]).flatten()
            length = len(self.experiment.species[item]['indices'])
            for observable in self.experiment.property_groups:
                path = join_path(item, observable)
                columns = self.experiment.property_groups[observable]
                structure[path] = {'indices': positions, 'columns': columns, 'length': length}

        return structure
