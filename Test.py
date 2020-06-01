import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import subprocess as sp
from scipy import signal
from scipy.optimize import curve_fit

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
        property_groups = ['Positions', 'Velocities', 'Forces']
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
        self.box_array = None

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

        self.box_array = [box_x, box_y, box_z]
        self.volume = box_x * box_y * box_z
        self.species = species_summary
        self.number_of_atoms = number_of_atoms
        self.properties = properties_summary

        print("Volume of the box is {0:8.3f} cubic Angstroms".format(self.volume))

    def Build_Database(self):

        # Create new analysis directory and change into it
        create_directory = sp.Popen(['mkdir {0}K_Analysis'.format(self.temperature)], shell=True)
        create_directory.wait()
        os.chdir('{0}K_Analysis'.format(self.temperature))

        data_array = self.Process_Input_File()  # Collect data array

        self.Get_System_Properties(data_array)  # Call function to analyse the system)

        class_array = []

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

    def Unwrap_Coordinates(self):
        """ Unwrap coordinates of trajectory

        For a number of properties the input data must in the form of unwrapped coordinates. This function takes the
        stored trajectory and returns the unwrapped coordinates so that they may be used for analysis.
        """
        os.chdir('{0}K_Analysis'.format(self.temperature))  # Change to correct directory
        box_array = [32.65, 32.65, 32.65]
        species_list = ['Na', 'Cl']
        positions_matrix = []
        for species in species_list:
            positions_matrix.append(np.load('{0}_Positions.npy'.format(species)))

        def Center_Box():
            """ Center atoms in box """

            for i in range(len(species_list)):
                for j in range(len(positions_matrix[0])):
                    positions_matrix[i][j] -= (box_array[0] / 2)

        Center_Box()

        print("\n --- Beginning to unwrap coordinates --- \n")

        for i in range(len(species_list)):
            for j in range(len(positions_matrix[i])):
                difference = np.diff(positions_matrix[i][j], axis=0)  # Difference between all atoms in the array

                # Indices where the atoms jump in the original array
                box_jump = [np.where(difference[:, 0] >= (box_array[0]/2))[0],
                            np.where(difference[:, 1] >= (box_array[1]/2))[0],
                            np.where(difference[:, 2] >= (box_array[2]/2))[0]]

                #Indices of first box cross
                box_cross = [box_jump[0] - 1, box_jump[1] - 1, box_jump[2] - 1]
                box_cross[0] = box_cross[0][:-1]
                box_cross[1] = box_cross[1][:-1]
                box_cross[2] = box_cross[2][:-1]

                for k in range(len(box_cross[0])):
                    positions_matrix[i][j][:, 0][box_cross[0][k]:] -= np.sign(difference[box_cross[0][k]][0])*0.5*box_array[0]
                for k in range(len(box_cross[1])):
                    positions_matrix[i][j][:, 1][box_cross[1][k]:] -= np.sign(difference[box_cross[1][k]][1])*0.5*box_array[1]
                for k in range(len(box_cross[2])):
                    positions_matrix[i][j][:, 2][box_cross[2][k]:] -= np.sign(difference[box_cross[2][k]][2])*0.5*box_array[2]

            np.save('{0}_Unwrapped.npy'.format(species_list[i]), positions_matrix[i])
            os.chdir('../')

    def Einstein_Diffusion_Coefficients(self):
        """ Calculate the Einstein self diffusion coefficients

            A function to implement the Einstein method for the calculation of the self diffusion coefficients
            of a liquid. In this method, unwrapped trajectories are read in and the MSD of the positions calculated and
            a gradient w.r.t time is calculated over several ranges to calculate and error measure.

            Data is loaded from the working directory.
        """

        os.chdir('{0}K_Analysis'.format(self.temperature)) # Change into analysis directory

        species_list = ['Na', 'Cl']
        positions_matrix = []
        for species in species_list:
            positions_matrix.append(np.load('{0}_Unwrapped.npy'.format(species)))

        msd_x = [0.00 for i in range(len(positions_matrix[1][0]))]
        msd_y = [0.00 for i in range(len(positions_matrix[1][0]))]
        msd_z = [0.00 for i in range(len(positions_matrix[1][0]))]

        print(positions_matrix[0][0])

        for i in range(len(positions_matrix[0])):
            msd_x += abs(positions_matrix[0][i][:, 0] - positions_matrix[0][i][0][0])**2
            msd_y += abs(positions_matrix[0][i][:, 1] - positions_matrix[0][i][1][0])**2
            msd_z += abs(positions_matrix[0][i][:, 2] - positions_matrix[0][i][2][0])**2

        msd = (1/3)*(1/len(positions_matrix[0]))*(msd_x + msd_y + msd_z)*(1E-20)
        x = (1E-12)*np.array([i for i in range(len(msd))])*0.002*100

        def func(x, a, b):
            return a*x + b

        popt, pcov = curve_fit(func, x[1000:], msd[1000:])
        print((popt[0]/6))

        plt.plot(x, msd)
        #plt.plot(x, func(x, *popt))
        plt.show()
        plt.loglog(x, msd)
        plt.show()

    def Green_Kubo_Diffusion_Coefficients(self):
        """ Calculate the Green_Kubo Diffusion coefficients

        Function to implement a Green-Kubo method for the calculation of diffusion coefficients whereby the velocity
        autocorrelation function is integrated over and divided by 3. Autocorrelation is performed using the scipy
        fft correlate function in order to speed up the calculation.

        Data is loaded from the working directory upon calling the function.

        velocity[i][j][:, k][l]

        i --> Atomic species
        j --> atom j
        k --> x, y or z
        l --> time step l
        Add class load functionality!!!!!!!!!!!!!
        """

        os.chdir('{0}K_Analysis'.format(self.temperature))  # Change to correct directory

        species_list = ['Na', 'Cl']
        velocity_matrix = []
        for species in species_list:
            velocity_matrix.append(np.load('{0}_Velocities.npy'.format(species)))

        time = (1E-12)*np.array([i for i in range(len(velocity_matrix[0][0]))])*(0.002)*3

        def Singular_Diffusion_Coefficients():
            vacf_a = np.zeros(2 * len(velocity_matrix[0][0]) - 1)
            vacf_b = np.zeros(2 * len(velocity_matrix[0][0]) - 1)

            for i in range(len(velocity_matrix[0])):
                vacf_a += (1/500)*np.array(
                    signal.correlate(velocity_matrix[0][i][:, 0], velocity_matrix[0][i][:, 0], mode='full',
                                     method='fft') +
                    signal.correlate(velocity_matrix[0][i][:, 1], velocity_matrix[0][i][:, 1], mode='full',
                                     method='fft') +
                    signal.correlate(velocity_matrix[0][i][:, 2], velocity_matrix[0][i][:, 2], mode='full',
                                     method='fft'))
                vacf_b += (1/500)*np.array(
                    signal.correlate(velocity_matrix[1][i][:, 0], velocity_matrix[1][i][:, 0], mode='full',
                                     method='fft') +
                    signal.correlate(velocity_matrix[1][i][:, 1], velocity_matrix[1][i][:, 1], mode='full',
                                     method='fft') +
                    signal.correlate(velocity_matrix[1][i][:, 2], velocity_matrix[1][i][:, 2], mode='full',
                                     method='fft'))

            sub_a = vacf_a
            sub_b = vacf_b
            vacf_a = (1/len(vacf_a))*vacf_a[int(len(vacf_a) / 2):]*(1E-20)/(1E-24)
            vacf_b = (1/len(vacf_b))*vacf_b[int(len(vacf_b) / 2):]*(1E-20)/(1E-24)

            D_a = np.trapz(vacf_a, x=time)/3
            D_b = np.trapz(vacf_b, x=time)/3
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
                vacf_a += (1/(499*498))*np.array(signal.correlate(velocity_matrix[0][i][:, 0], velocity_a[:, 0] - velocity_matrix[0][i][0][0], mode='full', method='fft') +
                               signal.correlate(velocity_matrix[0][i][:, 0], velocity_a[:, 1] - velocity_matrix[0][i][1][0], mode='full', method='fft') +
                               signal.correlate(velocity_matrix[0][i][:, 0], velocity_a[:, 2] - velocity_matrix[0][i][2][0], mode='full', method='fft'))
                vacf_b += (1/(499*498))*np.array(signal.correlate(velocity_matrix[1][i][:, 0], velocity_b[:, 0] - velocity_matrix[1][i][0][0], mode='full', method='fft') +
                               signal.correlate(velocity_matrix[1][i][:, 0], velocity_b[:, 1] - velocity_matrix[1][i][1][0], mode='full', method='fft') +
                               signal.correlate(velocity_matrix[1][i][:, 0], velocity_b[:, 2] - velocity_matrix[1][i][2][0], mode='full', method='fft'))
                vacf_c += (1/(500*500))*np.array(signal.correlate(velocity_b[:, 0], velocity_b[:, 0], mode='full', method='fft') +
                               signal.correlate(velocity_matrix[0][i][:, 0], velocity_b[:, 1], mode='full', method='fft') +
                               signal.correlate(velocity_matrix[0][i][:, 0], velocity_b[:, 2], mode='full', method='fft'))

            vacf_a = 1000*(1/(len(vacf_a)))*vacf_a[int(len(vacf_a) / 2):] * (1E-20) / (1E-24)
            vacf_b = 1000*(1/(len(vacf_b)))*vacf_b[int(len(vacf_b) / 2):] * (1E-20) / (1E-24)
            vacf_c = 1000*(1/(len(vacf_c)))*vacf_c[int(len(vacf_c) / 2):] * (1E-20) / (1E-24)

            print(velocity_matrix[0][0])
            print(velocity_matrix[0][0][0])
            print(velocity_matrix[0][0][0][0])

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

            os.chdir('../'.format(self.temperature))  # Change to correct directory

    def Nernst_Einstein_Conductivity(self):
        """ Calculate Nernst-Einstein Conductivity

        Function to determine the Nernst-Einstein as well as the corrected Nernst-Einstein
        conductivity of a system.
        """
        pass

    def Green_Kubo_Conductivity():

        os.chdir('{0}K_Analysis'.format(self.temperature))  # Change to correct directory

        species_list = ['Na', 'Cl']
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

        current = (1.60217662E-19)*(np.array(velocity_a) - np.array(velocity_b))

        jacf = (signal.correlate(current[:, 0], current[:, 0], mode='full', method='fft') +
                signal.correlate(current[:, 1], current[:, 1], mode='full', method='fft') +
                signal.correlate(current[:, 2], current[:, 2], mode='full', method='fft'))

        jacf = 1/(len(jacf))*(jacf[int((len(jacf)/2)):])*((1E-20) / (1E-24))

        sigma = (1/(3*(1300)*((32.28E-10)**3)*(1.38064852E-23)))*np.trapz(jacf, x = time)/100
        print(sigma)

        plt.plot(time, jacf)
        plt.show()

        os.chdir('../'.format(self.temperature))  # Change to correct directory

        Singular_Diffusion_Coefficients()
        Distinct_Diffusion_Coefficients()
        Green_Kubo_Conductivity()


def main():
    """ Main function to run processes """

    NaCl_1400K = Trajectory("/beegfs/work/stovey/LAMMPSSims/NaCl/scaledSim/Thesis_Sims/1450K/NaCl_Velocities.xyz", '1450K.npy', 1450)
    #NaCl_1400K.Build_Database()
    #NaCl_1400K.Unwrap_Coordinates()
    NaCl_1400K.Einstein_Diffusion_Coefficients()

if __name__ == "__main__":
    """ Standard python boilerplate """
    main()
