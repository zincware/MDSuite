"""
Author: Samuel Tovey
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Purpose: Larger methods used in the Trajectory class
"""
import Meta_Functions
import numpy as np


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


class Species_Properties_Methods:

    def Generate_LAMMPS_Property_Matrices(self):
        property_groups = Meta_Functions.Extract_LAMMPS_Properties(self.properties)
        property_list = list(self.properties)
        for i in range(len(property_groups)):
            saved_property = property_groups[i]
            temp = []
            for index in self.species_positions:
                temp.append(np.hstack([
                    np.array(self.data[index::1 * (self.number_of_atoms + 9)])[:,
                    self.properties[property_list[self.dimensions * i]]].astype(float)[:, None],
                    np.array(self.data[index::1 * (self.number_of_atoms + 9)])[:,
                    self.properties[property_list[self.dimensions * i + 1]]].astype(float)[:, None],
                    np.array(self.data[index::1 * (self.number_of_atoms + 9)])[:,
                    self.properties[property_list[self.dimensions * i + 2]]].astype(float)[:, None]]))
            np.save('{0}_{1}.npy'.format(self.species, saved_property), temp)

    def Generate_EXTXYZ_Property_Matrices(self):
        property_groups = Meta_Functions.Extract_extxyz_Properties(self.properties)
        property_list = list(self.properties)
        for i in range(len(property_groups)):
            saved_property = property_groups[i]
            temp = []
            for index in self.species_positions:
                temp.append(np.hstack([
                    np.array(self.data[index::1 * (self.number_of_atoms + 2)])[:,
                    1 + i*self.dimensions].astype(float)[:, None],
                    np.array(self.data[index::1 * (self.number_of_atoms + 2)])[:,
                    2 + i*self.dimensions].astype(float)[:, None],
                    np.array(self.data[index::1 * (self.number_of_atoms + 2)])[:,
                    3 + i*self.dimensions].astype(float)[:, None]]))
            np.save('{0}_{1}.npy'.format(self.species, saved_property), temp)


