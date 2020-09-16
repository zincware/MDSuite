"""
Author: Samuel Tovey
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Purpose: Larger methods used in the Trajectory class
"""
import mdsuite.Meta_Functions as Meta_Functions
import numpy as np
import h5py as hf

class Trajectory_Methods:
    """ Methods to be used in the Trajectory class """

    def Get_LAMMPS_Properties(self):
        """ Get the properties of the system from a custom lammps dump file

            args:
                data_array (list) -- Array containing trajectory data

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

        with open(self.filename) as f:
            head = [next(f).split() for i in range(9)]
            f.seek(0)  # Go back to the start of the file

            data_array = [next(f).split() for i in range(int(head[3][0]) + 9)]  # Get first configuration
            second_configuration = [next(f).split() for i in range(int(head[3][0]) + 9)] # Get the second

        
        # Calculate the number of atoms and configurations in the system
        number_of_atoms = int(data_array[3][0])

        number_of_lines = Meta_Functions.Line_Counter(self.filename)
        number_of_configurations = int(number_of_lines / (number_of_atoms + 9))
        batch_size = Meta_Functions.Optimize_Batch_Size(self.filename, number_of_configurations)

        time_0 = float(data_array[1][0])
        time_1 = float(second_configuration[1][0])
        sample_rate = time_1 - time_0
        time_N = (number_of_configurations - number_of_configurations % batch_size)*sample_rate

        # Find the information regarding species in the system and construct a dictionary
        for i in range(len(data_array[8])):
            if data_array[8][i] == "element":
                element_index = i - 2

        for i in range(9, number_of_atoms + 9):
            if data_array[i][element_index] not in species_summary:
                species_summary[data_array[i][element_index]] = []

            species_summary[data_array[i][2]].append(i)

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
        for item in list(self.species.keys()):
            database.create_group(item)
            for property in property_groups:
                database[item].create_group(property)
                database[item][property].create_dataset("x", (len(self.species[item]), self.number_of_configurations-
                                                              self.number_of_configurations % self.batch_size),
                                                        compression="gzip", compression_opts=9)
                database[item][property].create_dataset("y", (len(self.species[item]), self.number_of_configurations-
                                                              self.number_of_configurations % self.batch_size),
                                                        compression="gzip", compression_opts=9)
                database[item][property].create_dataset("z", (len(self.species[item]), self.number_of_configurations -
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
            positions = np.array([np.array(self.species[item]) + i * self.number_of_atoms - 9 for i in
                                  range(int(partitioned_configurations))]).flatten()
            for property in self.property_groups:
                database[item][property]["x"][:, counter:counter + partitioned_configurations] = \
                    data[positions][:, self.property_groups[property][0]].astype(float).reshape(
                        (len(self.species[item]), partitioned_configurations),
                        order='F')
                database[item][property]["y"][:, counter:counter + partitioned_configurations] = \
                    data[positions][:, self.property_groups[property][1]].astype(float).reshape(
                        (len(self.species[item]), partitioned_configurations),
                        order='F')
                database[item][property]["z"][:, counter:counter + partitioned_configurations] = \
                    data[positions][:, self.property_groups[property][2]].astype(float).reshape(
                        (len(self.species[item]), partitioned_configurations),
                        order='F')

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

        with open(f"{property}_generated.xyz", 'w') as f:
            for i in range(self.number_of_configurations):
                f.write(self.number_of_atoms)
                f.write("Generated by te mdsuite xyz writer")
                for j in range(len(species)):
                    for atom in data_matrix[j]:
                        f.write(f"{species[j]:<2}    {atom[i][0]:>9.4f}    {atom[i][1]:>9.4f}    {atom[i][2]:>9.4f}")


        # Construct the write array
        #for i in range(number_of_configurations):
        #    write_array.append(number_of_atoms)
        #    write_array.append("Nothing to see here")
        #    for j in range(len(data)):
        #        for k in range(len(data[j])):
        #            write_array.append("{0:<}    {1:>9.4f}    {2:>9.4f}    {3:>9.4f}".format(species[j],
                                                                                             #data[j][k][i][0],
                                                                                             #data[j][k][i][1],
                                                                                             #ata[j][k][i][2]))

        # Write the array to an output file
        #with open("output.xyz", "w") as f:
        #    for line in write_array:
        #       f.write("%s\n" % line)