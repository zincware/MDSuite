"""
Author: Samuel Tovey
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Purpose: Larger methods used in the Trajectory class
"""
import Meta_Functions


class Trajectory_Methods:
    """ Methods to be used in the Trajectory class """

    def Get_LAMMPS_Properties(self, data_array):
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

        # Calculate the number of atoms and configurations in the system
        number_of_atoms = int(data_array[3][0])
        number_of_configurations = int(len(data_array) / (number_of_atoms + 9))

        # Find the information regarding species in the system and construct a dictionary
        for i in range(9, number_of_atoms + 9):
            if data_array[i][2] not in species_summary:
                species_summary[data_array[i][2]] = []

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
        self.dimensions = Meta_Functions.Get_Dimensionality(box)
        self.box_array = box
        self.volume = box[0] * box[1] * box[2]
        self.species = species_summary
        self.number_of_atoms = number_of_atoms
        self.properties = properties_summary
        self.number_of_configurations = number_of_configurations

    def Get_EXTXYZ_Properties(self, data_array):
        """ Function to process extxyz input files """

        # Define necessary properties and attributes
        species_summary = {}
        properties_summary = {}
        extxyz_properties = {'pos', 'force'}

        number_of_atoms = int(data_array[0])


