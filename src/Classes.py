"""
Author: Samuel Tovey
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Purpose: Class functionality of the program
"""

import numpy as np
import os
import pandas as pd
import subprocess as sp
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
import Methods
import Meta_Functions
import pickle


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

    def __init__(self, species, number_of_atoms, data, species_positions, properties, dimensions, file_format):
        """ Standard class initialization """

        self.species = species
        self.number_of_atoms = number_of_atoms
        self.data = data
        self.species_positions = species_positions
        self.properties = properties
        self.dimensions = dimensions
        self.file_format = file_format

    def Generate_Property_Matrices(self):
        """ Create matrix of atom positions and save files

        mat = [[[atom_0_0(x,y,z)],[atom_0_1(x, y, z)], ..., [atom_0_i(x, y, z)]],
              ..., ...,
              [[atom_n_0(x,y,z)],[atom_n_1(x, y, z)], ..., [atom_n_i(x, y, z)]]]

        """

        if self.file_format == 'lammps':
            Methods.Species_Properties_Methods.Generate_LAMMPS_Property_Matrices(self)
        else:
            Methods.Species_Properties_Methods.Generate_EXTXYZ_Property_Matrices(self)


class Trajectory(Methods.Trajectory_Methods):
    """ Trajectory from simulation

    Class to structure and analyze the dump files from a LAMMPS simulation.
    Will calculate the following properties:
            - Green-Kubo Diffusion Coefficients
            - Nernst-Einstein Conductivity with corrections
            - Einstein Helfand Conductivity
            - Green Kubo Conductivity
    Future support for:
            - Radial Distribution Functions
            - Coordination Numbers
    """

    def __init__(self, analysis_name):
        """ Initialise with filename """

        self.filename = None
        self.analysis_name = analysis_name
        self.temperature = None
        self.volume = None
        self.species = None
        self.number_of_atoms = None
        self.properties = None
        self.dimensions = None
        self.box_array = None
        self.number_of_configurations = None
        self.singular_diffusion_coefficients = None
        self.distinct_diffusion_coefficients = None

    def From_User(self):
        """ Get system specific inputs

        If a new project is called, this function will gather extra data from the user
        """

        self.filename = input("File name: ")
        self.temperature = float(input("Temperature: "))

    def Save_Class(self):
        """ Saves class instance

        In order to keep properties of a class the state must be stored. This method will store the instance of the
        class for later re-loading
        """

        save_file = open("{0}.txt".format(self.analysis_name), 'wb')
        save_file.write(pickle.dumps(self.__dict__))
        save_file.close()

    def Load_Class(self):
        """ Load class instance

        A function to load a class instance given the project name.
        """
        os.chdir('../Project_Directories/{0}_Analysis'.format(self.analysis_name))

        class_file = open('{0}.txt'.format(self.analysis_name), 'rb')
        pickle_data = class_file.read()
        class_file.close()

        self.__dict__ = pickle.loads(pickle_data)

        os.chdir('../../src')

    def Process_Input_File(self):
        """ Process the input file

        returns:
            data_array (list) -- Array containing the trajectory data
        """

        data_array = []  # Initialize empty array for the data

        if self.filename[-6:] == 'extxyz':
            file_format = 'extxyz'
        else:
            file_format = 'lammps'

        # Store the file data in an array
        with open(self.filename) as f:
            for line in f:
                data_array.append(line.split())

        return data_array, file_format

    def Get_System_Properties(self, data_array, file_format):
        """ Get the properties of the system

        args:
            data_array (list) -- Array containing trajectory data
        """

        if file_format == 'lammps':
            Methods.Trajectory_Methods.Get_LAMMPS_Properties(self, data_array)
        else:
            Methods.Trajectory_Methods.Get_EXTXYZ_Properties(self, data_array)

    def Build_Database(self):
        """ Build the 'database' for the analysis

        Within this function, all properties directly present in the data are extracted and saved as .npy arrays
        in the project directory.
        """

        self.From_User() # Get additional information

        # Create new analysis directory and change into it
        create_directory = sp.Popen(['mkdir ../Project_Directories/{0}_Analysis'.format(self.analysis_name)],
                                    shell=True)
        create_directory.wait()
        os.chdir('../Project_Directories/{0}_Analysis'.format(self.analysis_name))

        data_array, file_format = self.Process_Input_File()  # Collect data array

        self.Get_System_Properties(data_array, file_format)  # Call function to analyse the system

        class_array = [] # Define empty array to store species classes

        # Instantiate a species class for each species
        for i in range(len(self.species)):
            class_array.append(
                Species_Properties(list(self.species)[i], self.number_of_atoms, data_array,
                                   self.species[list(self.species)[i]], self.properties, self.dimensions, file_format))

        # Generate the property matrices for each species
        for i in range(len(class_array)):
            class_array[i].Generate_Property_Matrices()

        del data_array
        self.Save_Class()
        os.chdir('../')

        print("\n ** Database has been constructed and saved for {0} ** \n".format(self.analysis_name))

    def Unwrap_Coordinates(self):
        """ Unwrap coordinates of trajectory

        For a number of properties the input data must in the form of unwrapped coordinates. This function takes the
        stored trajectory and returns the unwrapped coordinates so that they may be used for analysis.
        """
        os.chdir('../Project_Directories/{0}_Analysis'.format(self.analysis_name))  # Change to correct directory
        box_array = self.box_array
        species_list = list(self.species)
        positions_matrix = []
        for species in species_list:
            positions_matrix.append(np.load('{0}_Positions.npy'.format(species)))

        def Center_Box():
            """ Center atoms in box """

            for i in range(len(species_list)):
                for j in range(len(positions_matrix[0])):
                    positions_matrix[i][j] -= (box_array[0] / 2)

        def Unwrap():
            """ Unwrap the coordinates """

            Center_Box()  # Unwrap the coordinates

            print("\n --- Beginning to unwrap coordinates --- \n")

            for i in range(len(species_list)):
                for j in range(len(positions_matrix[0])):
                    difference = np.diff(positions_matrix[i][j], axis=0)  # Difference between all atoms in the array

                    # Indices where the atoms jump in the original array
                    box_jump = [np.where(abs(difference[:, 0]) >= (box_array[0] / 2))[0],
                                np.where(abs(difference[:, 1]) >= (box_array[1] / 2))[0],
                                np.where(abs(difference[:, 2]) >= (box_array[2] / 2))[0]]

                    # Indices of first box cross
                    box_cross = [box_jump[0] + 1, box_jump[1] + 1, box_jump[2] + 1]
                    box_cross[0] = box_cross[0]
                    box_cross[1] = box_cross[1]
                    box_cross[2] = box_cross[2]

                    for k in range(len(box_cross[0])):
                        positions_matrix[i][j][:, 0][box_cross[0][k]:] -= np.sign(difference[box_cross[0][k] - 1][0]) * \
                                                                          box_array[0]
                    for k in range(len(box_cross[1])):
                        positions_matrix[i][j][:, 1][box_cross[1][k]:] -= np.sign(difference[box_cross[1][k] - 1][1]) * \
                                                                          box_array[1]
                    for k in range(len(box_cross[2])):
                        positions_matrix[i][j][:, 2][box_cross[2][k]:] -= np.sign(difference[box_cross[2][k] - 1][2]) * \
                                                                          box_array[2]

                np.save('{0}_Unwrapped.npy'.format(species_list[i]), positions_matrix[i])
        Unwrap()
        print("\n --- Finished unwrapping coordinates --- \n")
        os.chdir('../')

    def Einstein_Diffusion_Coefficients(self):
        """ Calculate the Einstein self diffusion coefficients

            A function to implement the Einstein method for the calculation of the self diffusion coefficients
            of a liquid. In this method, unwrapped trajectories are read in and the MSD of the positions calculated and
            a gradient w.r.t time is calculated over several ranges to calculate and error measure.

            Data is loaded from the working directory.
        """

        os.chdir('../Project_Directories/{0}_Analysis'.format(self.analysis_name))  # Change into analysis directory

        species_list = list(self.species)
        positions_matrix = []
        for species in species_list:
            positions_matrix.append(np.load('{0}_Unwrapped.npy'.format(species)))

        def Singular_Diffusion_Coefficients(data):
            """ Calculate singular diffusion coefficients """

            # Define empty arrays
            msd_x = [0.00 for i in range(len(data[0]))]
            msd_y = [0.00 for i in range(len(data[0]))]
            msd_z = [0.00 for i in range(len(data[0]))]

            for i in range(len(positions_matrix[0])):
                msd_x += abs(data[i][:, 0] - data[i][0][0])
                msd_y += abs(data[i][:, 1] - data[i][0][1])
                msd_z += abs(data[i][:, 2] - data[i][0][2])

            msd = (1 / len(data)) * (msd_x ** 2 + msd_y ** 2 + msd_z ** 2) * (1E-20) * (1/3)
            time = (1E-12) * np.array([i for i in range(len(msd))]) * 0.002 * 3 * 100

            def func(time, a, b):
                return a * time + b

            popt, pcov = curve_fit(func, time[2000:], msd[2000:])
            print((popt[0] / 6))
            plt.plot(time, msd)
            plt.show()

        def Distinct_Diffusion_Coefficients(data):
            """ Calculate the Distinct Diffusion Coefficients """

            indices = [i for i in range(len(data))]
            correlations = list(combinations_with_replacement(indices, 2))

            coefficients = []  # Define empty coefficients list

            # Calculate the coefficients
            for i in range(len(correlations)):
                species_a = (1/len(data[correlations[i][0]]))*np.sum(data[correlations[i][0]], axis=0)
                species_b = (1/len(data[correlations[i][1]]))*np.sum(data[correlations[i][1]], axis=0)

                msd_x = []
                msd_y = []
                msd_z = []

                for j in range(len(species_a) - 1):
                    msd_x.append(np.pad(abs(species_a[j:, 0] - species_a[j][0])*abs(species_b[j:, 0] - species_b[j][0]),
                                        (0, j), constant_values=np.nan))
                    msd_y.append(np.pad(abs(species_a[j:, 1] - species_a[j][1])*abs(species_b[j:, 1] - species_b[j][1]),
                                        (0, j), constant_values=np.nan))
                    msd_z.append(np.pad(species_a[j:, 2] - species_a[j][2]*species_b[j:, 2] - species_b[j][2],
                                        (0, j), constant_values=np.nan))

                msd_x = np.nanmean(msd_x, axis=0)[:-100]
                msd_y = np.nanmean(msd_y, axis=0)[:-100]
                msd_z = np.nanmean(msd_z, axis=0)[:-100]

                msd = (1E-20) * (msd_x + msd_y + msd_z) * (1 / 3) * (len(data[correlations[i][0]]) +
                                                                     len(data[correlations[i][1]]))
                time = (1E-12) * 3 * 100 * np.array([i for i in range(len(msd))]) * 0.002

                plt.plot(time, msd)
                plt.show()

                def func(x, a, b):
                    return a * x + b

                popt, pcov = curve_fit(func, time, msd)
                coefficients.append((popt[0] / 6))
            print(coefficients)

        Distinct_Diffusion_Coefficients(positions_matrix)
        Singular_Diffusion_Coefficients(positions_matrix[0])
        Singular_Diffusion_Coefficients(positions_matrix[1])

    def Green_Kubo_Diffusion_Coefficients(self):
        """ Calculate the Green_Kubo Diffusion coefficients

        Function to implement a Green-Kubo method for the calculation of diffusion coefficients whereby the velocity
        autocorrelation function is integrated over and divided by 3. Autocorrelation is performed using the scipy
        fft correlate function in order to speed up the calculation.
        """

        os.chdir('../Project_Directories/{0}_Analysis'.format(self.analysis_name))  # Change to correct directory

        species_list = list(self.species)
        velocity_matrix = []
        for species in species_list:
            velocity_matrix.append(np.load('{0}_Velocities.npy'.format(species)))

        time = (1E-12) * np.array([i for i in range(len(velocity_matrix[0][0]))]) * (0.002) * 100 * 3

        def Singular_Diffusion_Coefficients():
            vacf_a = np.zeros(2 * len(velocity_matrix[0][0]) - 1)
            vacf_b = np.zeros(2 * len(velocity_matrix[0][0]) - 1)

            for i in range(len(velocity_matrix[0])):
                vacf_a += (1 / len(velocity_matrix[0])) * np.array(
                    signal.correlate(velocity_matrix[0][i][:, 0], velocity_matrix[0][i][:, 0], mode='full',
                                     method='fft') +
                    signal.correlate(velocity_matrix[0][i][:, 1], velocity_matrix[0][i][:, 1], mode='full',
                                     method='fft') +
                    signal.correlate(velocity_matrix[0][i][:, 2], velocity_matrix[0][i][:, 2], mode='full',
                                     method='fft'))
                vacf_b += (1 / len(velocity_matrix[1])) * np.array(
                    signal.correlate(velocity_matrix[1][i][:, 0], velocity_matrix[1][i][:, 0], mode='full',
                                     method='fft') +
                    signal.correlate(velocity_matrix[1][i][:, 1], velocity_matrix[1][i][:, 1], mode='full',
                                     method='fft') +
                    signal.correlate(velocity_matrix[1][i][:, 2], velocity_matrix[1][i][:, 2], mode='full',
                                     method='fft'))

            sub_a = vacf_a
            sub_b = vacf_b
            vacf_a = (1 / len(vacf_a)) * vacf_a[int(len(vacf_a) / 2):] * ((1E-20) / (1E-24))
            vacf_b = (1 / len(vacf_b)) * vacf_b[int(len(vacf_b) / 2):] * ((1E-20) / (1E-24))

            D_a = np.trapz(vacf_a, x=time) / 3
            D_b = np.trapz(vacf_b, x=time) / 3
            print(D_a)
            print(D_b)

            plt.plot(time, vacf_a)
            plt.plot(time, vacf_b)
            plt.show()

            return D_a, D_b, sub_a, sub_b

        def Distinct_Diffusion_Coefficients():

            vacf_a = np.zeros(2 * len(velocity_matrix[0][0]) - 1)
            vacf_b = np.zeros(2 * len(velocity_matrix[0][0]) - 1)
            vacf_c = np.zeros(2 * len(velocity_matrix[0][0]) - 1)

            velocity_a = []
            velocity_b = []
            for i in range(len(velocity_matrix[0][0])):
                velocity_a.append(np.sum(velocity_matrix[0][:, i], axis=0))
                velocity_b.append(np.sum(velocity_matrix[1][:, i], axis=0))

            velocity_a = np.array(velocity_a)
            velocity_b = np.array(velocity_b)

            for i in range(len(velocity_matrix[0])):
                vacf_a += (1 / (499 * 498)) * np.array(
                    signal.correlate(velocity_matrix[0][i][:, 0], velocity_a[:, 0] - velocity_matrix[0][i][0][0],
                                     mode='full', method='fft') +
                    signal.correlate(velocity_matrix[0][i][:, 0], velocity_a[:, 1] - velocity_matrix[0][i][1][0],
                                     mode='full', method='fft') +
                    signal.correlate(velocity_matrix[0][i][:, 0], velocity_a[:, 2] - velocity_matrix[0][i][2][0],
                                     mode='full', method='fft'))
                vacf_b += (1 / (499 * 498)) * np.array(
                    signal.correlate(velocity_matrix[1][i][:, 0], velocity_b[:, 0] - velocity_matrix[1][i][0][0],
                                     mode='full', method='fft') +
                    signal.correlate(velocity_matrix[1][i][:, 0], velocity_b[:, 1] - velocity_matrix[1][i][1][0],
                                     mode='full', method='fft') +
                    signal.correlate(velocity_matrix[1][i][:, 0], velocity_b[:, 2] - velocity_matrix[1][i][2][0],
                                     mode='full', method='fft'))
                vacf_c += (1 / (500 * 500)) * np.array(
                    signal.correlate(velocity_b[:, 0], velocity_b[:, 0], mode='full', method='fft') +
                    signal.correlate(velocity_matrix[0][i][:, 0], velocity_b[:, 1], mode='full', method='fft') +
                    signal.correlate(velocity_matrix[0][i][:, 0], velocity_b[:, 2], mode='full', method='fft'))

            vacf_a = self.number_of_atoms * (1 / (len(vacf_a))) * vacf_a[int(len(vacf_a) / 2):] * (1E-20) / (1E-24)
            vacf_b = self.number_of_atoms * (1 / (len(vacf_b))) * vacf_b[int(len(vacf_b) / 2):] * (1E-20) / (1E-24)
            vacf_c = self.number_of_atoms * (1 / (len(vacf_c))) * vacf_c[int(len(vacf_c) / 2):] * (1E-20) / (1E-24)

            D_a = np.trapz(vacf_a, x=time) / 6
            D_b = np.trapz(vacf_b, x=time) / 6
            D_c = np.trapz(vacf_c, x=time) / 6

            print(D_a)
            print(D_b)
            print(D_c)

            plt.plot(time, vacf_a)
            plt.plot(time, vacf_b)
            plt.plot(time, vacf_c)
            plt.show()

        _, _, _, _ = Singular_Diffusion_Coefficients()
        # Distinct_Diffusion_Coefficients()
        os.chdir('../'.format(self.temperature))  # Change to correct directory

    def Nernst_Einstein_Conductivity(self):
        """ Calculate Nernst-Einstein Conductivity

        A function to determine the Nernst-Einstein as well as the corrected Nernst-Einstein
        conductivity of a system.
        """
        pass

    def Green_Kubo_Conductivity(self):
        """ Calculate Green-Kubo Conductivity

        A function to use the current autocorrelation function to calculate the Green-Kubo ionic conductivity of the
        system being studied.
        """

        q = 1.60217662E-19 # Define elementary charge
        kb = 1.38064852E-23 # Define the Boltzmann constant
        os.chdir('../Project_Directories/{0}_Analysis'.format(self.analysis_name))  # Change to correct directory

        species_list = list(self.species)
        velocity_matrix = []
        for species in species_list:
            velocity_matrix.append(np.load('{0}_Velocities.npy'.format(species)))

        time = (1E-12) * np.array([i for i in range(len(velocity_matrix[0][0]))]) * (0.002) * 3

        vacf_a = np.zeros(2 * len(velocity_matrix[0][0]) - 1)
        vacf_b = np.zeros(2 * len(velocity_matrix[0][0]) - 1)
        vacf_c = np.zeros(2 * len(velocity_matrix[0][0]) - 1)

        velocity_a = []
        velocity_b = []
        for i in range(len(velocity_matrix[0][0])):
            velocity_a.append(np.sum(velocity_matrix[0][:, i], axis=0))
            velocity_b.append(np.sum(velocity_matrix[1][:, i], axis=0))

        current = q * (np.array(velocity_a) - np.array(velocity_b))

        jacf = (signal.correlate(current[:, 0], current[:, 0], mode='full', method='fft') +
                signal.correlate(current[:, 1], current[:, 1], mode='full', method='fft') +
                signal.correlate(current[:, 2], current[:, 2], mode='full', method='fft'))

        jacf = 1 / (len(jacf)) * (jacf[int((len(jacf) / 2)):]) * ((1E-20) / (1E-24))

        sigma = (3 * self.temperature * (self.volume * kb)) * np.trapz(jacf, x=time) / 100
        print(sigma)

        plt.plot(time, jacf)
        plt.show()
