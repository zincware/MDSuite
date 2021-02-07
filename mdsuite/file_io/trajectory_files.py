"""
Parent class for file processing

Summary
-------
"""

import abc
import os
from typing import TextIO

import h5py as hf
import numpy as np
from tqdm import tqdm

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

    def __init__(self, obj, header_lines, file_path):
        """
        Python constructor

        Parameters
        ----------
        obj : object
                Experiment class instance to add to.

        header_lines : int
                Number of header lines in the given file format.
        """

        super().__init__(obj, header_lines, file_path)  # fill the parent class

    def _read_header(self, f: TextIO, offset: int = 0):
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
                list of data in the header
        """

        # Skip the offset data
        for i in range(offset):
            f.readline()

        return [next(f).split() for _ in range(self.header_lines)]  # Get the first header

    def read_configurations(self, number_of_configurations: int, file_object: TextIO, skip: bool = True):
        """
        Read in a number of configurations from a file

        Parameters
        ----------
        skip : bool
                If true, the header lines will be skipped, if not, the returned data will include the headers.
        number_of_configurations : int
                Number of configurations to be read in.
        file_object : obj
                File object to be read from.

        Returns
        -------
        configuration data : np.array
                Data read in from the file object.
        """

        configurations_data = []  # Define the empty data array

        for i in range(number_of_configurations):

            if skip:
                # Skip header lines.
                for j in range(self.header_lines):
                    file_object.readline()

            # Read the data into the arrays.
            for k in range(self.project.number_of_atoms):
                configurations_data.append(file_object.readline().split())

        return np.array(configurations_data)

    def build_database_skeleton(self):
        """
        Build skeleton of the hdf5 database

        Gathers all of the properties of the system using the relevant functions. Following the gathering
        of the system properties, this function will read through the first configuration of the dataset, and
        generate the necessary database structure to allow for the following generation to take place. This will
        include the separation of species, atoms, and properties. For a full description of the data structure,
        look into the documentation.
        """

        # Set the length of the trajectory TODO: Add smaller "remainder" section to get the last parts of the trajectory
        initial_length = self.project.number_of_configurations - \
                         self.project.number_of_configurations % self.project.batch_size

        axis_names = ('x', 'y', 'z', 'xy', 'xz', 'yz', 'yx', 'zx', 'zy')

        # Build the database structure
        with hf.File(os.path.join(self.project.database_path, 'database.hdf5'), 'w', libver='latest') as database:

            # Loop over the different species.
            for item in self.project.species:
                database.create_group(item)  # create a hdf5 group in the database

                # Loop over the properties available from the simulation.
                for observable, columns in self.project.property_groups.items():

                    # Check if the property is scalar of vector to correctly structure the dataset
                    if len(columns) == 1:  # scalar
                        # Create dataset directly in the species group using extendable ds and scale offset compression.
                        database[item].create_dataset(observable, (len(self.project.species[item]['indices']),
                                                                   initial_length),
                                                      maxshape=(
                                                          len(self.project.species[item]['indices']), None),
                                                      scaleoffset=5)

                    else:  # vector or matrix (symmetric or not)
                        n_cols = len(columns)
                        database[item].create_group(observable)
                        for axis in axis_names[0:n_cols]:
                            database[item][observable].create_dataset(axis, (len(self.project.species[item]['indices']),
                                                                             initial_length),
                                                                      maxshape=(
                                                                          len(self.project.species[item]['indices']),
                                                                          None),
                                                                      scaleoffset=5)

    def build_file_structure(self):
        """
        Build a skeleton of the file so that the database class can process it correctly.
        """

        structure = {}  # define initial dictionary
        batch_size = int(self.project.batch_size)

        # Loop over species
        for item in self.project.species:
            positions = np.array([np.array(self.project.species[item]['indices']) + i * self.project.number_of_atoms -
                                  self.header_lines for i in range(batch_size)]).flatten()
            length = len(self.project.species[item]['indices'])
            for observable in self.project.property_groups:
                path = os.path.join(item, observable)
                columns = self.project.property_groups[observable]
                structure[path] = {'indices': positions, 'columns': columns, 'length': length}

        return structure
