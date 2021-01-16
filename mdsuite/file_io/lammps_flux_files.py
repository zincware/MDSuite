"""
Module for reading lammps trajectory files

Summary
-------
"""

import h5py as hf
import numpy as np
from tqdm import tqdm

from mdsuite.file_io.file_read import FileProcessor
# from .file_io_dict import lammps_flux
from mdsuite.utils.meta_functions import optimize_batch_size

lammps_flux = {
    "Temperature": ["temp"],
    "Time": ["time"],
    "Flux_Thermal": ['c_flux_thermal[1]', 'c_flux_thermal[2]', 'c_flux_thermal[3]']
}


class LAMMPSFluxFile(FileProcessor):
    """
    Child class for the lammps file reader to read Flux files from LAMMPS.

    Attributes
    ----------
    obj : object
            Experiment class instance to add to

    header_lines : int
            Number of header lines in the file format (lammps = 9)

    lammpstraj : str
            Path to the trajectory file.
    """

    def __init__(self, obj, header_lines=9, lammpstraj=None):
        """
        Python class constructor
        """

        super().__init__(obj, header_lines)  # fill the parent class
        self.lammpstraj = lammpstraj  # lammps file to read from.
        self.project.volume = None
        self.project.number_of_atoms = None

    def process_trajectory_file(self, update_class=True):
        """ Get additional information from the trajectory file

        In this method, there are several doc string styled comments. This is included as there are several components
        of the method that are all related to the analysis of the trajectory file.

        Parameters
        ----------
        update_class : bool
                Boolean decision on whether or not to update the class. If yes, the full saved class instance will be
                updated with new information. This is necessary on the first run of data addition to the database. After
                this point, when new data is added, this is no longer required as other methods will take care of
                updating the properties that change with new data. In fact, it will set the number of configurations to
                only the new data, which will be wrong.
        """

        n_lines_header = 0  # number of lines of header
        with open(self.project.trajectory_file) as f:
            header = []
            for line in f:
                n_lines_header += 1
                if line.startswith("#"):
                    header.append(line.split())
                else:
                    header_line = line.split()  # after the comments, we have the line with the variables
                    break

        self.header_lines = n_lines_header

        with open(self.project.trajectory_file) as f:
            number_of_configurations = sum(1 for _ in f) - n_lines_header

        # Find properties available for analysis
        column_dict_properties = self._get_column_properties(header_line)  # get column properties
        self.project.property_groups = self._extract_properties(lammps_flux,
                                                                column_dict_properties)  # Get the observable groups

        batch_size = optimize_batch_size(self.project.trajectory_file, number_of_configurations)

        # get time related properties of the system
        with open(self.project.trajectory_file) as f:
            # skip the header
            for _ in range(n_lines_header):
                next(f)
            time_0_line = f.readline().split()
            time_0 = float(time_0_line[column_dict_properties['time']])
            time_1_line = f.readline().split()
            time_1 = float(time_1_line[column_dict_properties['time']])

        sample_rate = (time_1 - time_0) / self.project.time_step
        time_n = (number_of_configurations - number_of_configurations % batch_size) * sample_rate

        # Update class attributes with calculated data
        self.project.batch_size = batch_size
        # self.properties = properties_summary
        self.project.number_of_configurations = number_of_configurations
        self.project.sample_rate = sample_rate
        self.time_0 = time_0

        # Get the number of atoms if not set in initialization
        if self.project.number_of_atoms is None:
            self.project.number_of_atoms = int(header[2][1])  # hopefully always in the same position

        # Get the volume, if not set in initialization
        if self.project.volume is None:
            self.project.volume = float(header[4][7])  # hopefully always in the same position

        self.project.species = {'1': []}

        if update_class:
            self.project.batch_size = batch_size

        else:
            self.project.batch_size = batch_size
            # return [1, 1, 1, number_of_configurations]

    @staticmethod
    def _get_column_properties(header_line):
        header_line = header_line[4:]
        properties_summary = {variable: idx + 2 for idx, variable in enumerate(header_line)}
        return properties_summary

    def build_database_skeleton(self):
        """
        We need to override the method because the flux files have a different structure
        """
        database = hf.File('{0}/{1}/{1}.hdf5'.format(self.project.storage_path, self.project.analysis_name), 'w',
                           libver='latest')
        axis_names = ('x', 'y', 'z')

        # Build the database structure
        database.create_group('1')
        for property_in, columns in self.project.property_groups.items():
            if len(columns) == 1:
                database['1'].create_dataset(property_in, (self.project.number_of_configurations -
                                                           self.project.number_of_configurations % self.project.batch_size,),
                                             compression="gzip", compression_opts=9)
            elif len(columns) == 3:
                database['1'].create_group(property_in)
                for axis in axis_names:
                    database['1'][property_in].create_dataset(axis, (self.project.number_of_configurations -
                                                                     self.project.number_of_configurations % self.project.batch_size,),
                                                              compression="gzip", compression_opts=9)

    def fill_database(self, counter=0):

        loop_range = int(
            (self.project.number_of_configurations - counter) / self.project.batch_size)  # loop range for the data.
        with hf.File("{0}/{1}/{1}.hdf5".format(self.project.storage_path, self.project.analysis_name),
                     "r+") as database:
            with open(self.project.trajectory_file) as f:
                for _ in tqdm(range(loop_range), ncols=70):
                    batch_data = self.read_configurations(self.project.batch_size, f)  # load the batch data
                    self.process_configurations(batch_data, database, counter)  # process the trajectory
                    counter += self.project.batch_size  # Update counter

    @staticmethod
    def _get_column_properties(header_line):
        properties_summary = {variable: idx for idx, variable in enumerate(header_line)}
        return properties_summary

    def read_configurations(self, number_of_configurations, file_object):
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
                database['1'][property_group][:] = data[:, columns[0]].astype(float)
            else:
                for column, axis in zip(columns, axis_names):
                    database['1'][property_group][axis][:] = data[:, column].astype(float)
