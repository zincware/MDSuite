import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import subprocess as sp
from scipy import signal

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
    
    def __init__(self, filename, analysis_name, temperature, volume):
        """ Intialize with filename """

        self.filename = filename
        self.analysis_name = analysis_name
        self.temperature = temperature
        self.volume = volume
    
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

        def __init__(self, species, number_of_atoms, data, data_range):
            """ Standard class initialization """

            self.species = species
            self.number_of_atoms = number_of_atoms
            self.data = data
            self.data_range = data_range

        def Generate_Property_Matrices(self):
            """ Create matrix of atom positions and save files

            mat = [[[atom_0_0(x,y,z)],[atom_0_1(x, y, z)], ..., [atom_0_i(x, y, z)]], 
                  ..., ...,
                  [[atom_n_0(x,y,z)],[atom_n_1(x, y, z)], ..., [atom_n_i(x, y, z)]]]

            """
            positions_matrix = []  # Only temporary, must soft code for arbitrary number of parameters
            velocity_matrix = []
            data_array = self.data

            for i in range(self.data_range[0], self.data_range[1]):
                #np.save(np.hstack([np.array(data_array[i::3*(self.number_of_atoms + 9)])[:, 3].astype(float),
                                        #np.array(data_array[i::3*(self.number_of_atoms + 9)])[:, 4].astype(float),
                                        #np.array(data_array[i::3*(self.number_of_atoms + 9)])[:, 5].astype(float)]))
                velocity_matrix.apend([np.array(data_array[i::2*(self.number_of_atoms + 9)])[:, 6].astype(float),
                                        np.array(data_array[i::2*(self.number_of_atoms + 9)])[:, 7].astype(float),
                                        np.array(data_array[i::2*(self.number_of_atoms + 9)])[:, 8].astype(float)])
            return velocity_matrix

    def Process_Input_File(self):
        """ Process the input file 
        
        returns:
            data_array (list) -- Array containing the trajectory data
        """

        data_array = [] # Initialize empty array for the data

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
        LAMMPS_Properties = ['x', 'y', 'z', 'xs', 'ys', 'zs', 'xu', 'yu', 'zu', 'xsu', 'ysu', 'zsu', 'ix', 'iy', 'iz',
                            'vx', 'vy', 'vz', 'fx', 'fy', 'fz', 'mux', 'muy', 'muz', 'mu', 'omegax', 'omegay', 'omegaz',
                            'angmomx', 'angmomy', 'angmomz', 'tqx', 'tqy', 'tqz']

        number_of_atoms = int(data_array[3][0]) # Get number of atoms in system
        number_of_configurations = len(data_array)/(number_of_atoms + 9) # Get the number of configurations from system
        
        species_summary = {} # Define an empty dictionary for the species information
        properties_summary = {}

        # Find the information regarding species in the system
        for i in range(9, number_of_atoms + 9):
            if data_array[i][2] not in species_summary: # Add new species to dictionary
                species_summary[data_array[i][2]] = 0

            species_summary[data_array[i][2]] += 1 # Sum over all instances of the species in configurations
        
        # Find properties available for analysis
        for i in range(len(data_array[8])):
            if data_array[8][i] in LAMMPS_Properties: # Check if property is in the LAMMPS property array
                properties_summary[data_array[8][i]] = i

        return species_summary, properties_summary, number_of_atoms

    def Build_Database(self, data_array):

        species, properties, number_of_atoms = self.Get_System_Properties(data_array)

        class_array = []
        for i in range(len(species)):
            class_array.append(self.Species_Properties(list(species)[i], number_of_atoms, data_array, [9 + i*500, 508 + i*500]))
        
        velocities = [class_array[0].Generate_Property_Matrices(), class_array[1].Generate_Property_Matrices()]
        del data_array

        print("\n --- Database has been Constructed, Proceeding with analysis for {0}K --- \n".format(self.temperature))

        return velocities
    
    def Green_Kubo_Diffusion(self, velocities):
        """ Calculate the Green Kubo Diffusion """

        def Single_Diffusion_Coefficients():
            """ Calculate single diffusion coefficients """

            vacf_a = np.array([0.0 for i in range(66667)])
            vacf_b = np.array([0.0 for i in range(66667)])
            print("\n--- Beginning Single Diffusion Coefficient Calculations ---\n")
            for i in range(len(velocities[0])):

                vacf_a += (1/500)*np.array(signal.correlate(velocities[0][i][0], velocities[0][i][0], mode = 'full', method = 'fft') +
                                signal.correlate(velocities[0][i][1], velocities[0][i][1], mode = 'full', method = 'fft') + 
                                signal.correlate(velocities[0][i][2], velocities[0][i][2], mode = 'full', method = 'fft')) 
                vacf_b += (1/500)*np.array(signal.correlate(velocities[1][i][0], velocities[1][i][0], mode = 'full', method = 'fft') +
                                signal.correlate(velocities[1][i][1], velocities[1][i][1], mode = 'full', method = 'fft') +
                                signal.correlate(velocities[1][i][2], velocities[1][i][2], mode = 'full', method = 'fft'))
            
            vacf_a = (1/66667)*vacf_a[int(len(vacf_a)/2):]*(1E-20)/(1E-24)
            vacf_b = (1/66667)*vacf_b[int(len(vacf_b)/2):]*(1E-20)/(1E-24)

            time = np.array([i for i in range(len(vacf_a))])*(1E-12)*0.005

            plt.plot(time, vacf_a)
            plt.show()
            plt.plot(time, vacf_b)
            plt.show()

            D_a = np.trapz(vacf_a, x = time)/3
            D_b = np.trapz(vacf_b, x = time)/3

            print("Single Diffusion Coefficient Na: {0}\n".format(D_a))
            print("Single Diffusion Coefficient Cl: {0}\n".format(D_b))

            np.save("{0}_vacf_s_a.npy".format(self.temperature), vacf_a)
            np.save("{0}_vacf_s_b.npy".format(self.temperature), vacf_b)

            return D_a, D_b


        def Distinct_Diffusion_Coefficients():
            """ Calculate distinct diffusion coefficients """

            reference_array_a = [np.sum(velocities[0][:][0], axis=0), np.sum(velocities[0][:][1], axis=0), np.sum(velocities[0][:][2], axis=0)]
            reference_array_b = [np.sum(velocities[1][:][0], axis=0), np.sum(velocities[1][:][1], axis=0), np.sum(velocities[1][:][2], axis=0)]
            print("\n--- Begining Distinct Diffusion Coefficient Calculations ---\n")
            vacf_a = np.array([0.0 for i in range(66667)])
            vacf_b = np.array([0.0 for i in range(66667)])
            vacf_c = np.array([0.0 for i in range(66667)])
            print(len(velocities[0][0][0]))

            for i in range(len(velocities[0])):

                vacf_a += (1000/(499*500))*np.array(signal.correlate(velocities[0][i][0], reference_array_a[0] - velocities[0][i][0], mode = 'full', method = 'fft') +
                                signal.correlate(velocities[0][i][1], reference_array_a[1] - velocities[0][i][1], mode = 'full', method = 'fft') + 
                                signal.correlate(velocities[0][i][2], reference_array_a[2] - velocities[0][i][2], mode = 'full', method = 'fft'))
                vacf_b += (1000/(499*500))*np.array(signal.correlate(velocities[1][i][0], reference_array_b[0] - velocities[1][i][0], mode = 'full', method = 'fft') +
                                signal.correlate(velocities[1][i][1], reference_array_b[1] - velocities[1][i][1], mode = 'full', method = 'fft') + 
                                signal.correlate(velocities[1][i][2], reference_array_b[2] - velocities[1][i][2], mode = 'full', method = 'fft'))
                vacf_c += (1000/(499*500))*np.array(signal.correlate(velocities[0][i][0], reference_array_b[0] - velocities[0][i][0], mode = 'full', method = 'fft') +
                                signal.correlate(velocities[0][i][1], reference_array_b[1]- velocities[0][i][1], mode = 'full', method = 'fft') + 
                                signal.correlate(velocities[0][i][2], reference_array_b[2]- velocities[0][i][2], mode = 'full', method = 'fft'))
            
            vacf_a = (1/66667)*np.array(vacf_a[int(len(vacf_a)/2):]*(1E-20)/(1E-24))
            vacf_b = (1/66667)*np.array(vacf_b[int(len(vacf_b)/2):]*(1E-20)/(1E-24))
            vacf_c = (1/66667)*np.array(vacf_c[int(len(vacf_c)/2):]*(1E-20)/(1E-24))

            time = np.array([i for i in range(len(vacf_a))])*(1E-12)*0.005

            plt.plot(time, vacf_a)
            plt.show()
            plt.plot(time, vacf_b)
            plt.show()
            plt.plot(time, vacf_c)
            plt.show()

            D_a = np.trapz(vacf_a, x = time)/6
            D_b = np.trapz(vacf_b, x = time)/6
            D_c = np.trapz(vacf_c, x = time)/6

            print("Distinct Diffusion Coefficient Na: {0}\n".format(D_a))
            print("Distinct Diffusion Coefficient Cl: {0}\n".format(D_b))
            print("Distinct Diffusion Coefficient NaCl: {0}\n".format(D_c))

        
            np.save("{0}_vacf_d_a.npy".format(self.temperature), vacf_a)
            np.save("{0}_vacf_d_b.npy".format(self.temperature), vacf_b)
            np.save("{0}_vacf_d_c.npy".format(self.temperature), vacf_c)

            return D_a, D_b, D_c

        Ds_a, Ds_b = Single_Diffusion_Coefficients()
        Dd_a, Dd_b, Dd_c = Distinct_Diffusion_Coefficients()

        return [Ds_a, Ds_b, Dd_a, Dd_b, Dd_c]
    
    def Nernst_Einstein_Conductivity(self, diffusion_array):
        """ Calculate the Nernst-Einstein conductivities

        Function to calculate the Nernst-Einstein as well as the corrected
        Nernst-Einstein conductivities given the diffusion coefficients and 
        the distinct diffusion coefficients as calculated in the Green-Kubo 
        conductivities method

        args:
            diffusion_array (list) -- [Diffs_a, Diffs_b, Diffd_a, Diffd_b, Diffd_c]
        """

        q = 1.6E-19
        kb = 1.38E-23
        beta = 1/(kb*self.temperature)
        rho = 1000/self.volume

        def Conductivity():
            """ Calculate direct Nernst-Einstein """

            sigma = (q**2)*(rho)*(beta)*(diffusion_array[0] + diffusion_array[1])/2
            
            print("Ionic Conductivity: {0}\n".format(sigma/100))


        def Corrected_Conductvity():
            """ Calculate corrected Nernst-Einstein """

            sigma = 0.5*(q**2)*rho*beta*(diffusion_array[0] + diffusion_array[1] + 
                                                    0.5*abs(diffusion_array[2]) + 
                                                    0.5*abs(diffusion_array[3]) - 
                                                    abs(diffusion_array[4]))
            
            print("Corrected Ionic Conductivity: {0}\n".format(sigma/100))
            
        Conductivity()
        Corrected_Conductvity()

    def Einstein_Helfand_Conductivity(self, positions):
        """ Calculate Einstein Helfand Conductivity

        Determine the ionic conductivity of the system using the
        Einstein Helfand dipole Mean square dispalcement method

        args:
            positions (array) --  Matrix of atom positions through time.
        """



    def Green_Kubo_Conductivity(self, velocitities):
        """ Calculate Green-Kubo Conductivity

        Calculate the ionic conductivtiy of the system using the 
        current autocorrelation function as in the Green-Kubo
        approach.
        """
        
        # Assign some constants
        q = 1.0
        kb = 1.38064852E-23
        beta = 1/(kb*self.temperature)
        V = self.volume

        # Sum over all velocities in confgurations
        current_x = q*(np.sum(velocitities[0][:][0][:], axis = 0) - np.sum(velocitities[1][:][0][:], axis = 0))
        current_y = q*(np.sum(velocitities[0][:][1][:], axis = 0) - np.sum(velocitities[1][:][1][:], axis = 0))
        current_z = q*(np.sum(velocitities[0][:][2][:], axis = 0) - np.sum(velocitities[1][:][2][:], axis = 0))

        print(len(current_x))

        Jacf = (signal.correlate(current_x, current_x, mode = 'same', method = 'fft') + 
                signal.correlate(current_y, current_y, mode = 'same', method = 'fft') +
                signal.correlate(current_z, current_z, mode = 'same', method = 'fft'))
        
        Jacf = Jacf[int(len(Jacf)/2):]*((1E-20)/(1E-24))
        time = np.array([i for i in range(len(Jacf))])*(1E-12)*0.0005

        plt.plot(time, Jacf)
        plt.show()

        time_1 = np.array([i for i in range(len(current_x))])*0.0005
        save_array =[time_1[:, None], current_x[:, None], current_y[:, None], current_z[:, None]]
        data = np.hstack(save_array)
        np.save('1300K_Velocities.npy', data.astype(np.float32))

        #np.save("{0}_Jacf.npy".format(self.temperature), Jacf)
        #time = np.array([i for i in range(len(current_x))])*0.002


        #sigma = (1/3)*beta*(1/V)*np.trapz(Jacf, x = time)
        #print("Einstein-Helfan Conductivity: {0}\n".format(sigma))


def main():
    """ Main function to run processes """

    #LiF_1300K = Trajectory("/beegfs/work/stovey/LAMMPSSims/LiF/smallSim/LiF_Pos_Vel_Frc.xyz", '1400K.npy', 1300, (10.6E-10)**3)
    #data_array = LiF_1300K.Process_Input_File()
    #velocities = LiF_1300K.Build_Database(data_array)
    #LiF_1300K.Green_Kubo_Conductivity(velocities)
    #Diffusion = LiF_1300K.Green_Kubo_Diffusion(velocities)
    #LiF_1300K.Nernst_Einstein_Conductivity(Diffusion)

    # LiF_1300K = Trajectory("/beegfs/work/stovey/LAMMPSSims/LiF/smallSim/LiF_Pos_Vel_Frc.xyz", '1400K.npy', 1300, (10.6E-10)**3)
    # data_array = LiF_1300K.Process_Input_File()
    # velocities = LiF_1300K.Build_Database(data_array)
    # LiF_1300K.Green_Kubo_Conductivity(velocities)
    # Diffusion = LiF_1300K.Green_Kubo_Diffusion(velocities)
    # LiF_1300K.Nernst_Einstein_Conductivity(Diffusion)


    print("\n############### Finished Analysis ###############\n")


if __name__ == "__main__":
    """ Standard python boilerplate """
    main()

