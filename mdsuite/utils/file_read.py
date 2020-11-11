""" Parent class for file processing """

import numpy as np

import h5py as hf


class FileProcessor:
    """ Parent class for file reading and processing """

    def __init__(self, obj, header_lines):
        """ Python constructor """

        self.project = obj
        self.header_lines = header_lines

    def _read_configurations(self, number_of_configurations, file_object):
        """ Read in a configuration from a txt file"""

        configurations_data = []

        for i in range(number_of_configurations):
            # Skip header lines
            for j in range(self.header_lines):
                file_object.readline()

            for k in range(self.project.number_of_atoms):
                configurations_data.append(file_object.readline().split())

        return np.array(configurations_data)

    def _extract_properties(self):
        """ Get property groups from the trajectory """

        raise NotImplementedError("Implemented in child class")  # Raise error if this class method is called directly

    def _process_trajectory_file(self):
        """ Get property groups from the trajectory """

        raise NotImplementedError("Implemented in child class")  # Raise error if this class method is called directly

    def _process_log_file(self):
        """ Get property groups from the trajectory """

        raise NotImplementedError("Implemented in child class")  # Raise error if this class method is called directly

    def _build_database_skeleton(self):
        """ Build skeleton of the hdf5 database

            Gathers all of the properties of the system using the relevant functions. Following the gathering
            of the system properties, this function will read through the first configuration of the dataset, and
            generate the necessary database structure to allow for the following generation to take place. This will
            include the separation of species, atoms, and properties. For a full description of the data structure,
            look into the documentation.
        """

        self.project.property_groups = self._extract_properties()  # Get the observable groups

        initial_length = self.project.number_of_configurations - self.project.number_of_configurations % self.project.batch_size

        # Build the database structure
        with hf.File('{0}/{1}/{1}.hdf5'.format(self.project.storage_path, self.project.analysis_name), 'w',
                     libver='latest') as database:
            for item in self.project.species:
                database.create_group(item)
                for observable in self.project.property_groups:
                    database[item].create_group(observable)
                    database[item][observable].create_dataset("x", (len(self.project.species[item]['indices']),
                                                                    initial_length),
                                                              maxshape=(
                                                              len(self.project.species[item]['indices']), None),
                                                              scaleoffset=5)

                    database[item][observable].create_dataset("y", (len(self.project.species[item]['indices']),
                                                                    initial_length),
                                                              maxshape=(
                                                              len(self.project.species[item]['indices']), None),
                                                              scaleoffset=5)

                    database[item][observable].create_dataset("z", (len(self.project.species[item]['indices']),
                                                                    initial_length),
                                                              maxshape=(
                                                              len(self.project.species[item]['indices']), None),
                                                              scaleoffset=5)

    def _resize_database(self):
        """ Resize the database skeleton """

        resize_factor = self.project.number_of_configurations - self.project.number_of_configurations % self.project.batch_size

        with hf.File('{0}/{1}/{1}.hdf5'.format(self.project.storage_path, self.project.analysis_name), 'r+',
                     libver='latest') as database:

            for species in self.project.species:
                for observable in self.project.property_groups:
                    database[species][observable]['x'].resize(resize_factor, 1)
                    database[species][observable]['y'].resize(resize_factor, 1)
                    database[species][observable]['z'].resize(resize_factor, 1)

    def _process_configurations(self, data, database, counter):
        """ Process the available data

                Called during the main database creation. This function will calculate the number of configurations
                within the raw data and process it.

                args:
                    data (numpy array) -- Array of the raw data for N configurations.
                    database (object) --
                    counter (int) --
                """

        # Re-calculate the number of available configurations for analysis
        partitioned_configurations = int(len(data) / self.project.number_of_atoms)

        for item in self.project.species:
            # get the new indices for the positions
            positions = np.array([np.array(self.project.species[item]['indices']) + i * self.project.number_of_atoms -
                                  self.header_lines for i in range(int(partitioned_configurations))]).flatten()
            # Fill the database
            for property_group in self.project.property_groups:
                database[item][property_group]["x"][:, counter:counter + partitioned_configurations] = \
                    data[positions][:, self.project.property_groups[property_group][0]].astype(float).reshape(
                        (len(self.project.species[item]['indices']), partitioned_configurations), order='F')

                database[item][property_group]["y"][:, counter:counter + partitioned_configurations] = \
                    data[positions][:, self.project.property_groups[property_group][1]].astype(float).reshape(
                        (len(self.project.species[item]['indices']), partitioned_configurations), order='F')

                database[item][property_group]["z"][:, counter:counter + partitioned_configurations] = \
                    data[positions][:, self.project.property_groups[property_group][2]].astype(float).reshape(
                        (len(self.project.species[item]['indices']), partitioned_configurations), order='F')
