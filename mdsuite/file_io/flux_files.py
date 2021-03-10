"""
Module for reading lammps trajectory files

Summary
-------
"""
import os

import h5py as hf
import numpy as np
from tqdm import tqdm

from mdsuite.file_io.file_read import FileProcessor
# from .file_io_dict import lammps_flux
from mdsuite.utils.meta_functions import optimize_batch_size


class FluxFile(FileProcessor):
    """
    Child class for the lammps file reader to read Flux files.

    Attributes
    ----------
    obj : object
            Experiment class instance to add to

    header_lines : int
            Number of header lines in the file format (lammps = 9)

    file_path : str
            Path to the trajectory file.
    """

    def __init__(self, obj, header_lines=9, file_path=None, sort: bool = False):
        """
        Python class constructor
        """

        super().__init__(obj, header_lines, file_path)  # fill the parent class
        self.project.volume = None
        self.project.number_of_atoms = None
        self.sort = sort

    def build_database_skeleton(self):
        """
        We need to override the method because the flux files have a different structure
        """
        # database = hf.File('{0}/{1}/{1}.hdf5'.format(self.project.storage_path, self.project.analysis_name), 'w',
        #                    libver='latest')
        axis_names = ('x', 'y', 'z', 'xy', 'xz', 'yz', 'yx', 'zx', 'zy')

        with hf.File(os.path.join(self.project.database_path, 'database.hdf5'), 'w', libver='latest') as database:
            # Build the database structure
            database.create_group('1')
            for property_in, columns in self.project.property_groups.items():
                if len(columns) == 1:
                    database['1'].create_dataset(property_in, (self.project.number_of_configurations -
                                                               self.project.number_of_configurations %
                                                               self.project.batch_size,),
                                                 compression="gzip", compression_opts=9)
                else:
                    n_cols = len(columns)
                    database['1'].create_group(property_in)
                    for axis in axis_names[0:n_cols]:
                        database['1'][property_in].create_dataset(axis, (self.project.number_of_configurations -
                                                                         self.project.number_of_configurations %
                                                                         self.project.batch_size,),
                                                                  compression="gzip", compression_opts=9)

    def fill_database(self, counter=0):
        """

        Parameters
        ----------
        counter
        """
        # loop range for the data.
        loop_range = int((self.project.number_of_configurations - counter) / self.project.batch_size)
        skip_header = 0
        with hf.File(os.path.join(self.project.database_path, 'database.hdf5'), "r+") as database:
            with open(self.project.trajectory_file) as f:
                for _ in tqdm(range(loop_range), ncols=70):
                    if skip_header == 0:
                        batch_data = self.read_configurations(self.project.batch_size, f,
                                                              skip=True)  # load the batch data
                    else:
                        batch_data = self.read_configurations(self.project.batch_size, f)  # load the batch data
                    self.process_configurations(batch_data, database, counter)  # process the trajectory
                    skip_header = 1  # turn off the header skip
                    counter += len(batch_data)  # Update counter

    def read_configurations(self, number_of_configurations, file_object, skip=False):
        """
        Read in a number of configurations from a file file

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

        if skip:
            # Skip header lines.
            [file_object.readline() for _ in range(self.header_lines)]

        for i in range(number_of_configurations):
            # Read the data into the arrays.
            configurations_data.append(file_object.readline().split())

        return np.array(configurations_data)

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

        """
        Fill the database
        """
        axis_names = ('x', 'y', 'z', 'xy', 'xz', 'yz')
        # Fill the database
        for property_group, columns in self.project.property_groups.items():
            num_columns = len(columns)
            if num_columns == 1:
                database['1'][property_group][counter:counter + len(data)] = data[:, columns[0]].astype(float)
            else:
                for column, axis in zip(columns, axis_names):
                    database['1'][property_group][axis][counter:counter + len(data)] = data[:, column].astype(float)
