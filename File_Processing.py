import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import subprocess as sp
from scipy import signal


class Species_Properties:
    """ Properties of a single species

    This class will conatin all the information about a single molecule
    present in the files being analyzed. Within this class there
    are n matrices corresponding to the n properties saved in the dump
    command from LAMMPS. For multi atom molecules, extra programming is required.

    args:
        species (str) -- molecule the class corresponds to
        number_of_atoms (int) -- Number of molecules in each configuration
        data (array) (list) -- Data array from which the information is taken
        data_range (list) -- Indices of all the molecules in each configurations

    """

    def __init__(self, species, number_of_atoms, data, data_range, properties, dimensions):
        """ Standard class initialization """

        self.species = species
        self.number_of_atoms = number_of_atoms
        self.data = data
        self.data_range = data_range
        self.properties = properties
        self.dimensions = dimensions

    def Generate_Property_Matrices(self):
        """ Create matrix of atom positions and save files

        mat = [[[atom_0_0(x,y,z)],[atom_0_1(x, y, z)], ..., [atom_0_i(x, y, z)]],
              ..., ...,
              [[atom_n_0(x,y,z)],[atom_n_1(x, y, z)], ..., [atom_n_i(x, y, z)]]]

        """

        #property_groups = self.properties / self.dimensions  # Determine how many properties are present by dimension
        property_groups = ['Positions', 'Velocities', 'Forces']  # Jan! Add you custom measurement here manually!! This will enable a new file to be made
        property_list = list(self.properties)
        for i in range(len(property_groups)):
            saved_property = property_groups[i]
            temp = []
            for j in range(self.data_range[0], self.data_range[1]):
                temp.append(np.hstack([
                    np.array(self.data[j::3 * (self.number_of_atoms + 9)])[:,
                    self.properties[property_list[self.dimensions * i]]].astype(float)[:, None],
                    np.array(self.data[j::3 * (self.number_of_atoms + 9)])[:,
                    self.properties[property_list[self.dimensions * i + 1]]].astype(float)[:, None],
                    np.array(self.data[j::3 * (self.number_of_atoms + 9)])[:,
                    self.properties[property_list[self.dimensions * i + 2]]].astype(float)[:, None]]))
            np.save('{0}_{1}.npy'.format(self.species, saved_property), temp)

class Trajectory:
    """ Trajectory from simulation

    Class to structure and analyze the dump files from a LAMMPS simulation.
    Will calculate the following properties:
            - Green-Kubo Diffusion Coefficients
            - Nerst-Einstein Conductivity with corrections
            - Einstein Helfand Conductivity
            - Green Kubo Conductivity
    Future support for:
            - Radial Distribution Functions
            - Coordination Numbers
    """

    def __init__(self, filename, analysis_name, temperature):
        """ Initialise with filename """

        self.filename = filename
        self.analysis_name = analysis_name
        self.temperature = temperature
        self.volume = None
        self.species = None
        self.number_of_atoms = None
        self.properties = None
        self.dimensions = None

    def Process_Input_File(self):
        """ Process the input file

        returns:
            data_array (list) -- Array containing the trajectory data
        """

        data_array = []  # Initialize empty array for the data

        # Store the file data in an array
        with open(self.filename) as f:
            for line in f:
                data_array.append(line.split())

        return data_array

    def Get_System_Properties(self, data_array):
        """ Get the properties of the system

        args:
            data_array (list) -- Array containing trajectory data

        returns:
            species_summary (dict) -- Dictionary containing all the species in the systems
                                      and how many of them there are in each configuration.
            properties_summary (dict) -- All the properties availabe in the dump file for
                                         analysis and their index in the file
        """

        # Stored properties avaiable in a LAMMPS dump file
        LAMMPS_Properties = {'x', 'y', 'z', 'xs', 'ys', 'zs', 'xu', 'yu', 'zu', 'xsu', 'ysu', 'zsu', 'ix', 'iy', 'iz',
                             'vx', 'vy', 'vz', 'fx', 'fy', 'fz', 'mux', 'muy', 'muz', 'mu', 'omegax', 'omegay',
                             'omegaz',
                             'angmomx', 'angmomy', 'angmomz', 'tqx', 'tqy', 'tqz'}

        number_of_atoms: int = int(data_array[3][0])  # Get number of atoms in system
        number_of_configurations: int = int(
            len(data_array) / (number_of_atoms + 9))  # Get the number of configurations from system

        species_summary = {}  # Define an empty dictionary for the species information
        properties_summary = {}  # Define empty dictionary for the simulation properties

        # Find the information regarding species in the system
        for i in range(9, number_of_atoms + 9):
            if data_array[i][2] not in species_summary:  # Add new species to dictionary
                species_summary[data_array[i][2]] = 0

            species_summary[data_array[i][2]] += 1  # Sum over all instances of the species in configurations

        # Find properties available for analysis
        for i in range(len(data_array[8])):
            if data_array[8][i] in LAMMPS_Properties:  # Check if property is in the LAMMPS property array
                properties_summary[data_array[8][i]] = i - 2

        # Get the box size from the system (In Angstroms for this investigation)

        box_x = (float(data_array[5][1][:-10]) - float(data_array[5][0][:-10])) * 10
        box_y = (float(data_array[6][1][:-10]) - float(data_array[6][0][:-10])) * 10
        box_z = (float(data_array[7][1][:-10]) - float(data_array[7][0][:-10])) * 10

        if box_x == 0 or box_y == 0 or box_z == 0.0:
            self.dimensions = 2
        elif box_x == 0 and box_y == 0 or box_x == 0 and box_z == 0 or box_y == 0 and box_z == 0:
            self.dimensions = 1
        else:
            self.dimensions = 3

        self.volume = box_x * box_y * box_z
        self.species = species_summary
        self.number_of_atoms = number_of_atoms
        self.properties = properties_summary

        print("Volume of the box is {0:8.3f} cubic Angstroms".format(self.volume))

    def Build_Database(self):
        """ Build database for use 
        
        Jan! - You will need to change the hardcoded parameters in the Species_Properties call on line 182
        """

        # Create new analysis directory and change into it
        create_directory = sp.Popen(['mkdir {0}K_Analysis'.format(self.temperature)], shell=True)
        create_directory.wait()
        os.chdir('{0}K_Analysis'.format(self.temperature))

        data_array = self.Process_Input_File()  # Collect data array

        self.Get_System_Properties(data_array)  # Call function to analyse the system)

        class_array = []

        print(self.species)

        print(self.species)
        for i in range(len(self.species)):
            class_array.append(
                Species_Properties(list(self.species)[i], self.number_of_atoms, data_array,
                                   [9 + i * 500, 508 +
                                    i * 500], self.properties, self.dimensions))

        for i in range(len(class_array)):
            class_array[i].Generate_Property_Matrices()

        del data_array
        os.chdir('../')

        print("\n --- Database has been Constructed, Proceeding with analysis for {0}K --- \n".format(self.temperature))


def main():
    """ Main function to run processes """

    NaCl_1400K = Trajectory("/beegfs/work/stovey/LAMMPSSims/NaCl/scaledSim/Thesis_Sims/1300K/rerun/NaCl_Velocities.xyz", '1400K.npy', 1300)
    NaCl_1400K.Build_Database()
    
if __name__ == "__main__":
    """ Standard python boilerplate """
    main()
