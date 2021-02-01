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

    def _read_header(self, f: TextIO, offset: int=0):
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

    def read_configurations(self, number_of_configurations, file_object, skip=True):
        """
        Read in a number of configurations from a file

        Parameters
        ----------
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

        axis_names = ('x', 'y', 'z', 'xy', 'xz', 'yz')

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

                    elif len(columns) == 6:  # symmetric tensor (for stress tensor for example)
                        database[item].create_group(observable)
                        for axis in axis_names:
                            database[item][observable].create_dataset(axis, (len(self.project.species[item]['indices']),
                                                                             initial_length),
                                                                      maxshape=(
                                                                          len(self.project.species[item]['indices']),
                                                                          None),
                                                                      scaleoffset=5)

                    else:  # vector
                        database[item].create_group(observable)
                        for axis in axis_names[0:3]:
                            database[item][observable].create_dataset(axis, (len(self.project.species[item]['indices']),
                                                                             initial_length),
                                                                      maxshape=(
                                                                          len(self.project.species[item]['indices']),
                                                                          None),
                                                                      scaleoffset=5)

    def resize_database(self):
        """
        Resize the database skeleton.

        """
        # Get the number of additional configurations TODO: Again add support for collecting the remainder.
        resize_factor = self.project.number_of_configurations - \
                        self.project.number_of_configurations % \
                        self.project.batch_size

        # Open the database and resize the database.
        with hf.File(os.path.join(self.project.database_path, 'database.hdf5'), 'r+',
                     libver='latest') as database:

            # Loop over species in the database.
            for species in self.project.species:

                # Loop over property being added to  and resize the datasets.
                for observable in self.project.property_groups:
                    database[species][observable]['x'].resize(resize_factor, 1)
                    database[species][observable]['y'].resize(resize_factor, 1)
                    database[species][observable]['z'].resize(resize_factor, 1)

    def fill_database(self, counter=0):
        """
        Loads data into a hdf5 database

        Parameters
        ----------
        counter : int
                Number of configurations that have been read in.
        """

        # loop range for the data.
        loop_range = int((self.project.number_of_configurations - counter) / self.project.batch_size)
        with hf.File(os.path.join(self.project.database_path, 'database.hdf5'), "r+") as database:
            with open(self.project.trajectory_file) as f:
                for _ in tqdm(range(loop_range), ncols=70):
                    batch_data = self.read_configurations(self.project.batch_size, f)  # load the batch data
                    self.process_configurations(batch_data, database, counter)  # process the trajectory
                    counter += self.project.batch_size  # Update counter

    def process_configurations(self, data, database, counter):
        """
        Process the available data

        Called during the main database creation. This function will calculate the number of configurations
        within the raw data and process it.

        Parameters
        ----------
        data : np.array
                Array of the raw data for N configurations.

        database : object
                Database in which to store the data.

        counter : int
                Which configuration to start from.
        """

        # Re-calculate the number of available configurations for analysis
        partitioned_configurations = int(len(data) / self.project.number_of_atoms)

        for item in self.project.species:
            """
            Get the new indices for the positions. This function requires the atoms to be in the same position during
            each configuration. The calculation simply adds multiples of the number of atoms and configurations to the
            position of each atom in order to read the correct part of the file.
            """
            # TODO: Implement a sort algorithm or something of the same kind.
            positions = np.array([np.array(self.project.species[item]['indices']) + i * self.project.number_of_atoms -
                                  self.header_lines for i in range(int(partitioned_configurations))]).flatten()

            """
            Fill the database
            """
            axis_names = ('x', 'y', 'z', 'xy', 'xz', 'yz')
            # Fill the database
            for property_group, columns in self.project.property_groups.items():
                num_columns = len(columns)
                if num_columns == 1:
                    database[item][property_group][:, counter:counter + partitioned_configurations] = \
                        data[positions][:, columns[0]].astype(float).reshape(
                            (len(self.project.species[item]['indices']), partitioned_configurations), order='F')
                else:
                    for column, axis in zip(columns, axis_names):
                        database[item][property_group][axis][:, counter:counter + partitioned_configurations] = \
                            data[positions][:, column].astype(float).reshape(
                                (len(self.project.species[item]['indices']), partitioned_configurations), order='F')
