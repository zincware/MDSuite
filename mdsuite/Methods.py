"""
Author: Samuel Tovey
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Purpose: Larger methods used in the Trajectory class
"""
import pickle

import mdsuite.Meta_Functions as Meta_Functions
import numpy as np
import h5py as hf
import mendeleev

class Trajectory_Methods:
    """ Methods to be used in the Trajectory class """

    def Get_LAMMPS_Properties(self):
        """ Get the properties of the system from a custom lammps dump file

            returns:
                species_summary (dict) -- Dictionary containing all the species in the systems
                                          and how many of them there are in each configuration.
                properties_summary (dict) -- All the properties available in the dump file for
                                             analysis and their index in the file
        """

        # Define necessary properties and attributes
        species_summary = {}
        properties_summary = {}
        LAMMPS_Properties_labels = {'x', 'y', 'z',
                             'xs', 'ys', 'zs',
                             'xu', 'yu', 'zu',
                             'xsu', 'ysu', 'zsu',
                             'ix', 'iy', 'iz',
                             'vx', 'vy', 'vz',
                             'fx', 'fy', 'fz',
                             'mux', 'muy', 'muz', 'mu',
                             'omegax', 'omegay', 'omegaz',
                             'angmomx', 'angmomy', 'angmomz',
                             'tqx', 'tqy', 'tqz'}

        nlines_header_block = 9
        with open(self.filename) as f:
            head = [next(f).split() for i in range(nlines_header_block)]
            f.seek(0)  # Go back to the start of the file
            # Calculate the number of atoms and configurations in the system
            number_of_atoms = int(head[3][0])
            # Get first configuration
            data_array = [next(f).split() for i in range(number_of_atoms + nlines_header_block)]  # Get first configuration
            second_configuration = [next(f).split() for i in range(number_of_atoms + nlines_header_block)] # Get the second

        number_of_lines = Meta_Functions.Line_Counter(self.filename)
        number_of_configurations = int(number_of_lines / (number_of_atoms + nlines_header_block)) # n of timesteps
        batch_size = Meta_Functions.Optimize_Batch_Size(self.filename, number_of_configurations)

        time_0 = float(data_array[1][0])
        time_1 = float(second_configuration[1][0])
        sample_rate = time_1 - time_0
        time_N = (number_of_configurations - number_of_configurations % batch_size)*sample_rate

        # Get the position of the element keyword so that any format can be given
        for i in range(len(data_array[8])):
            if data_array[8][i] == "element":
                element_index = i - 2

        # Find the information regarding species in the system and construct a dictionary
        for i in range(9, number_of_atoms + 9):
            if data_array[i][element_index] not in species_summary:
                species_summary[data_array[i][element_index]] = {}
                species_summary[data_array[i][element_index]]['indices'] = []

            species_summary[data_array[i][element_index]]['indices'].append(i)

        # Find properties available for analysis
        for i in range(len(data_array[8])):
            if data_array[8][i] in LAMMPS_Properties_labels:
                properties_summary[data_array[8][i]] = i - 2

        # Get the box size from the system
        box = [(float(data_array[5][1][:-10]) - float(data_array[5][0][:-10])) * 10,
               (float(data_array[6][1][:-10]) - float(data_array[6][0][:-10])) * 10,
               (float(data_array[7][1][:-10]) - float(data_array[7][0][:-10])) * 10]

        # Update class attributes with calculated data
        self.batch_size = batch_size
        self.dimensions = Meta_Functions.Get_Dimensionality(box)
        self.box_array = box
        self.volume = box[0] * box[1] * box[2]
        self.species = species_summary
        self.number_of_atoms = number_of_atoms
        self.properties = properties_summary
        self.number_of_configurations = number_of_configurations
        self.time_dimensions = [0.0, time_N*self.time_step*self.time_unit]
        self.sample_rate = sample_rate

    def Get_EXTXYZ_Properties(self, data_array):
        """ Function to process extxyz input files """

        print("This functionality does not currently work")
        return

        # Define necessary properties and attributes
        species_summary = {}
        properties_summary = {}
        extxyz_properties_keywords = ['pos', 'force']

        number_of_atoms = int(data_array[0][0])
        number_of_configurations = len(data_array) / (number_of_atoms + 2)
        box = [float(data_array[1][0][9:]), float(data_array[1][4]), float(data_array[1][8][:-1])]

        for i in range(2, number_of_atoms + 2):
            if data_array[i][0] not in species_summary:
                species_summary[data_array[i][0]] = []
            species_summary[data_array[i][0]].append(i)

        for i in range(len(extxyz_properties_keywords)):
            if extxyz_properties_keywords[i] in data_array[1][9]:
                properties_summary[extxyz_properties_keywords[i]] = 0

        self.dimensions = Meta_Functions.Get_Dimensionality(box)
        self.box_array = box
        self.volume = box[0] * box[1] * box[2]
        self.species = species_summary
        self.number_of_atoms = number_of_atoms
        self.properties = properties_summary
        self.number_of_configurations = number_of_configurations

    def Build_Species_Dictionary(self):
        """ Add information to the species dictionary

        A fundamental part of this package is species specific analysis. Therefore, the mendeleev python package is
        used to add important species specific information to the class. This will include the charge of the ions which
        will be used in conductivity calculations.

        returns:
            This method will update the class attributes in place and therefore, will not return anything explicitly.
        """

        for element in self.species:
            temp = mendeleev.element(element)

            charge = [] # Define empty charge array
            for ir in temp.ionic_radii:
                if ir.most_reliable is not True:
                    continue
                else:
                    charge.append(ir.charge)

            if not temp.ionic_radii is True:
                self.species[element]['charge'] = 0
            elif len(charge) == 0:
                self.species[element]['charge'] = [temp.ionic_radii[0].charge] # Case where most_reliable is all False
            elif all(elem == charge[0] for elem in charge) is True:
                self.species[element]['charge'] = [charge[0]]
            else:
                self.species[element]['charge'] = charge

            mass = []
            for iso in temp.isotopes:
                mass.append(iso.mass)
            self.species[element]['mass'] = mass

    def Build_Database_Skeleton(self):
        """ Build skeleton of the hdf5 database

        Gathers all of the properties of the system using the relevant functions. Following the gathering
        of the system properties, this function will read through the first configuration of the dataset, and
        generate the necessary database structure to allow for the following generation to take place. This will
        include the separation of species, atoms, and properties. For a full description of the data structure,
        look into the documentation.
        """

        database = hf.File('{0}/{1}/{1}.hdf5'.format(self.filepath, self.analysis_name), 'w', libver='latest')

        property_groups = Meta_Functions.Extract_LAMMPS_Properties(self.properties)  # Get the property groups
        self.property_groups = property_groups

        # Build the database structure
        for item in self.species:
            database.create_group(item)
            for property in property_groups:
                database[item].create_group(property)
                database[item][property].create_dataset("x", (len(self.species[item]['indices']), self.number_of_configurations-
                                                              self.number_of_configurations % self.batch_size),
                                                        compression="gzip", compression_opts=9)
                database[item][property].create_dataset("y", (len(self.species[item]['indices']), self.number_of_configurations-
                                                              self.number_of_configurations % self.batch_size),
                                                        compression="gzip", compression_opts=9)
                database[item][property].create_dataset("z", (len(self.species[item]['indices']), self.number_of_configurations -
                                                              self.number_of_configurations % self.batch_size),
                                                        compression="gzip", compression_opts=9)

    def Read_Configurations(self, N, f):
        """ Read in N configurations

        This function will read in N configurations from the file that has been opened previously by the parent method.

        args:

            N (int) -- Number of configurations to read in. This will depend on memory availability and the size of each
                        configuration. Automatic setting of this variable is not yet available and therefore, it will be set
                        manually.
            f (obj) --
        """

        data = []

        for i in range(N):
            # Skip header lines
            for j in range(9):
                f.readline()

            for k in range(self.number_of_atoms):
                data.append(f.readline().split())

        return np.array(data)

    def Process_Configurations(self, data, database, counter):
        """ Process the available data

        Called during the main database creation. This function will calculate the number of configurations within the
        raw data and process it.

        args:
            data (numpy array) -- Array of the raw data for N configurations.
            database (object) --
            counter (int) --
        """

        # Re-calculate the number of available configurations for analysis
        partitioned_configurations = int(len(data) / self.number_of_atoms)

        for item in self.species:
            # get the new indices for the positions
            positions = np.array([np.array(self.species[item]['indices']) + i * self.number_of_atoms - 9 for i in
                                  range(int(partitioned_configurations))]).flatten()
            # Fill the database
            for property_group in self.property_groups:
                database[item][property_group]["x"][:, counter:counter + partitioned_configurations] = \
                    data[positions][:, self.property_groups[property_group][0]].astype(float).reshape(
                        (len(self.species[item]['indices']), partitioned_configurations), order='F')

                database[item][property_group]["y"][:, counter:counter + partitioned_configurations] = \
                    data[positions][:, self.property_groups[property_group][1]].astype(float).reshape(
                        (len(self.species[item]['indices']), partitioned_configurations), order='F')

                database[item][property_group]["z"][:, counter:counter + partitioned_configurations] = \
                    data[positions][:, self.property_groups[property_group][2]].astype(float).reshape(
                        (len(self.species[item]['indices']), partitioned_configurations), order='F')

    def Print_Data_Structrure(self):
        """ Print the data structure of the hdf5 dataset """

        database = hf.File("{0}/{1}/{1}.hdf5".format(self.filepath, self.analysis_name), "r")

    def Write_XYZ(self, property="Positions", species=None):
        """ Write an xyz file from database array

        For some of the properties calculated it is beneficial to have an xyz file for analysis with other platforms.
        This function will write an xyz file from a numpy array of some property. Can be used in the visualization of
        trajectories.

        kwargs:
            property (str) -- Which property would you like to print
            species (list) -- List of species for which you would like to write the file
        """

        if species == None:
            species = list(self.species.keys())

        data_matrix = self.Load_Matrix(property, species)

        with open(f"{self.filepath}/{self.analysis_name}/{property}_{'_'.join(species)}.xyz", 'w') as f:
            for i in range(self.number_of_configurations):
                f.write(f"{self.number_of_atoms}\n")
                f.write("Generated by the mdsuite xyz writer\n")
                for j in range(len(species)):
                    for atom in data_matrix[j]:
                        f.write(f"{species[j]:<2}    {atom[i][0]:>9.4f}    {atom[i][1]:>9.4f}    {atom[i][2]:>9.4f}\n")

    def Save_Class(self):
        """ Saves class instance

        In order to keep properties of a class the state must be stored. This method will store the instance of the
        class for later re-loading
        """

        save_file = open("{0}/{1}/{1}.bin".format(self.filepath, self.analysis_name), 'wb')
        save_file.write(pickle.dumps(self.__dict__))
        save_file.close()

    def Load_Class(self):
        """ Load class instance

        A function to load a class instance given the project name.
        """

        class_file = open('{0}/{1}/{1}.bin'.format(self.filepath, self.analysis_name), 'rb')
        pickle_data = class_file.read()
        class_file.close()

        self.__dict__ = pickle.loads(pickle_data)

    def Print_Class_Attributes(self):
        """ Print all attributes of the class """

        attributes = []
        for item in vars(self).items():
            attributes.append(item)
        for tuple in attributes:
            print(f"{tuple[0]}: {tuple[1]}")

        return attributes