"""
Parent class for file processing

Summary
-------
"""

import h5py as hf
import numpy as np
from tqdm import tqdm
import abc

class FileProcessor(metaclass=abc.ABCMeta):
    """
    Parent class for file reading and processing

    Attributes
    ----------
    obj, project : object
            File object to be opened and read in.
    header_lines : int
            Number of header lines in the file format being read.
    """

    def __init__(self, obj, header_lines):
        """
        Python constructor

        Parameters
        ----------
        obj : object
                Experiment class instance to add to.

        header_lines : int
                Number of header lines in the given file format.
        """

        self.project = obj  # Experiment class instance to add to.
        self.header_lines = header_lines  # Number of header lines in the given file format.

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

        for i in range(number_of_configurations):

            # Skip header lines.
            for j in range(self.header_lines):
                file_object.readline()

            # Read the data into the arrays.
            for k in range(self.project.number_of_atoms):
                configurations_data.append(file_object.readline().split())

        return np.array(configurations_data)

    @abc.abstractmethod
    def process_trajectory_file(self):
        """
        Get property groups from the trajectory
        """

        raise NotImplementedError("Implemented in child class")  # Raise error if this class method is called directly


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
        with hf.File('{0}/{1}/{1}.hdf5'.format(self.project.storage_path, self.project.analysis_name), 'w',
                     libver='latest') as database:

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
                                                      scaleoffset=10)

                    elif len(columns) == 6:  # symmetric tensor (for stress tensor for example)
                        database[item].create_group(observable)
                        for axis in axis_names:
                            database[item][observable].create_dataset(axis, (len(self.project.species[item]['indices']),
                                                                             initial_length),
                                                                      maxshape=(
                                                                          len(self.project.species[item]['indices']),
                                                                          None),
                                                                      scaleoffset=10)

                    else:  # vector
                        database[item].create_group(observable)
                        for axis in axis_names[0:3]:
                            database[item][observable].create_dataset(axis, (len(self.project.species[item]['indices']),
                                                                             initial_length),
                                                                      maxshape=(
                                                                          len(self.project.species[item]['indices']),
                                                                          None),
                                                                      scaleoffset=10)

    def resize_database(self):
        """
        Resize the database skeleton.

        """

        # Get the number of additional configurations TODO: Again add support for collecting the remainder.
        resize_factor = self.project.number_of_configurations - \
                        self.project.number_of_configurations % \
                        self.project.batch_size

        # Open the database and resize the database.
        with hf.File('{0}/{1}/{1}.hdf5'.format(self.project.storage_path, self.project.analysis_name), 'r+',
                     libver='latest') as database:

            # Loop over species in the database.
            for species in self.project.species:

                # Loop over property being added to  and resize the datasets.
                for observable in self.project.property_groups:
                    database[species][observable]['x'].resize(resize_factor, 1)
                    database[species][observable]['y'].resize(resize_factor, 1)
                    database[species][observable]['z'].resize(resize_factor, 1)

    @abc.abstractmethod
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



    @staticmethod
    def _extract_properties(database_correspondance_dict, column_dict_properties):
        """
        Construct generalized property array

        Takes the lammps properties dictionary and constructs and array of properties which can be used by the species
        class.

        agrs:
            properties_dict (dict) -- A dictionary of all the available properties in the trajectory. This dictionary is
            built only from the LAMMPS symbols and therefore must be again processed to extract the useful information.

        returns:
            trajectory_properties (dict) -- A dictionary of the keyword labelled properties in the trajectory. The
            values of the dictionary keys correspond to the array location of the specific piece of data in the set.
        """

        # for each property label (position, velocity,etc) in the lammps definition
        for property_label, property_names in database_correspondance_dict.items():
            # for each coordinate for a given property label (position: x, y, z), get idx and the name
            for idx, property_name in enumerate(property_names):
                if property_name in column_dict_properties.keys():  # if this name (x) is in the input file properties
                    # we change the lammps_properties_dict replacing the string of the property name by the column name
                    database_correspondance_dict[property_label][idx] = column_dict_properties[property_name]

        # trajectory_properties only needs the labels with the integer columns, then we one copy those
        trajectory_properties = {}
        for property_label, properties_columns in database_correspondance_dict.items():
            if all([isinstance(property_column, int) for property_column in properties_columns]):
                trajectory_properties[property_label] = properties_columns

        print("I have found the following properties with the columns in []: ")
        [print(key, value) for key, value in trajectory_properties.items()]

        return trajectory_properties

    def fill_database(self, counter=0):
        """
        Loads data into a hdf5 database

        Parameters
        ----------
        trajectory_reader : object
                Instance of a trajectory reader class.

        counter : int
                Number of configurations that have been read in.
        """

        loop_range = int(
            (self.project.number_of_configurations - counter) / self.project.batch_size)  # loop range for the data.
        with hf.File("{0}/{1}/{1}.hdf5".format(self.project.storage_path, self.project.analysis_name),
                     "r+") as database:
            with open(self.project.trajectory_file) as f:
                for _ in tqdm(range(loop_range), ncols=70):
                    batch_data = self.read_configurations(self.project.batch_size, f)  # load the batch data
                    self.process_configurations(batch_data, database, counter)  # process the trajectory
                    counter += self.project.batch_size  # Update counter
