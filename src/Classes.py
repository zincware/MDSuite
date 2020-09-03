"""
Author: Samuel Tovey
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Purpose: Class functionality of the program
"""

import numpy as np
import os
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import Methods
import pickle
import h5py as hf
from alive_progress import alive_bar
import Constants
import Meta_Functions
import itertools


class Trajectory(Methods.Trajectory_Methods):
    """ Trajectory from simulation

    Attributes:

        filename (str) -- filename of the trajectory

        analysis_name (str) -- name of the analysis being performed e.g. NaCl_1400K

        new_project (bool) -- If the project has already been build, if so, the class state will be loaded

        filepath (str) -- where to store the data (best to have  drive capable of storing large files)

        temperature (float) -- temperature of the system

        time_step (float) -- time step in the simulation e.g 0.002

        time_unit (float) -- scaling factor for time, should result in the time being in SI units (seconds)
                             e.g. 1e-12 for ps

        volume (float) -- volume of the system

        species (dict) -- dictionary of the species in the system and their indices in the trajectory. e.g.
                          {'Na': [1, 3, 7, 9], 'Cl': [0, 2, 4, 5, 6, 8]}

        number_of_atoms (int) -- number of atoms in the system

        properties (dict) -- properties in the trajectory available for analysis, not important for understanding

        property_groups (list) -- property groups, e.g ["Forces", "Positions", "Velocities", "Torques"]

        dimensions (float) -- dimensionality of the system e.g. 3.0

        box_array (list) -- box lengths, e.g [10.7, 10.7, 10.8]

        number_of_configurations (int) -- number of configurations in the trajectory

        time_dimensions (list) -- Time domain in the system, e.g, for a 1ns simulation, [0.0, 1e-9]

        singular_diffusion_coefficient (dict) -- Dictionary of diffusion coefficients e.g. {'Na': 1e-8, 'Cl': 0.9e-8}

        distinct_diffusion_coefficients (dict) -- Dictionary of distinct diffusion coefficients
                                                  e.g. {'Na': 1, 'Cl': 0.5, 'NaCl': 0.9'}

        ionic_conductivity (float) -- Ionic conductivity of the system e.g. 4.5 S/cm
    """

    def __init__(self, analysis_name, new_project=False, storage_path=None,
                 temperature=None, time_step=None, time_unit=None, filename=None):
        """ Initialise with filename """

        self.filename = filename
        self.analysis_name = analysis_name
        self.new_project = new_project
        self.filepath = storage_path
        self.temperature = temperature
        self.time_step = time_step
        self.time_unit = time_unit
        self.sample_rate = None
        self.batch_size = None
        self.volume = None
        self.species = None
        self.number_of_atoms = None
        self.properties = None
        self.property_groups = None
        self.dimensions = None
        self.box_array = None
        self.number_of_configurations = None
        self.time_dimensions = None
        self.singular_diffusion_coefficients = None
        self.distinct_diffusion_coefficients = None
        self.ionic_conductivity = None

        if self.new_project == False:
            self.Load_Class()
        else:
            self.Build_Database()

    def Save_Class(self):
        """ Saves class instance

        In order to keep properties of a class the state must be stored. This method will store the instance of the
        class for later re-loading
        """

        save_file = open("{0}/{1}/{1}.txt".format(self.filepath, self.analysis_name), 'wb')
        save_file.write(pickle.dumps(self.__dict__))
        save_file.close()

    def Load_Class(self):
        """ Load class instance

        A function to load a class instance given the project name.
        """

        class_file = open('{0}/{1}/{1}.txt'.format(self.filepath, self.analysis_name), 'rb')
        pickle_data = class_file.read()
        class_file.close()

        self.__dict__ = pickle.loads(pickle_data)

    def Print_Class_Attributes(self):
        """ Print all attributes of the class """

        print(', '.join("%s: %s" % item for item in vars(self).items()))

    def Process_Input_File(self):
        """ Process the input file

        A trivial function to get the format of the input file. Will probably become more useful when we add support
        for more file formats.
        """

        if self.filename[-6:] == 'extxyz':
            file_format = 'extxyz'
        else:
            file_format = 'lammps'

        return file_format

    def Get_System_Properties(self, file_format):
        """ Get the properties of the system

        This method will call the Get_X_Properties depending on the file format. This function will update all of the
        class attributes and is necessary for the operation of the Build database method.

        args:
            file_format (str) -- Format of the file being read
        """

        if file_format == 'lammps':
            Methods.Trajectory_Methods.Get_LAMMPS_Properties(self)
        else:
            Methods.Trajectory_Methods.Get_EXTXYZ_Properties(self)

    def Build_Database(self):
        """ Build the 'database' for the analysis

        A method to build the database in the hdf5 format. Within this method, several other are called to develop the
        database skeleton, get configurations, and process and store the configurations. The method is accompanied
        by a loading bar which should be customized to make it more interesting.
        """

        # Create new analysis directory and change into it
        os.mkdir('{0}/{1}'.format(self.filepath, self.analysis_name))

        file_format = self.Process_Input_File()  # Collect data array
        self.Get_System_Properties(file_format)  # Update class attributes
        Methods.Trajectory_Methods.Build_Database_Skeleton(self)

        print("Beginning Build database")

        with hf.File("{0}/{1}/{1}.hdf5".format(self.filepath, self.analysis_name), "r+") as database:
            with open(self.filename) as f:
                counter = 0
                with alive_bar(int(self.number_of_configurations / self.batch_size), bar='blocks', spinner='dots') as bar:
                    for i in range(int(self.number_of_configurations / self.batch_size)):
                        test = Methods.Trajectory_Methods.Read_Configurations(self, self.batch_size, f)

                        Methods.Trajectory_Methods.Process_Configurations(self, test, database, counter)

                        counter += self.batch_size
                        bar()

        self.Save_Class()

        print("\n ** Database has been constructed and saved for {0} ** \n".format(self.analysis_name))

    def Unwrap_Coordinates(self):
        """ Unwrap coordinates of trajectory

        For a number of properties the input data must in the form of unwrapped coordinates. This function takes the
        stored trajectory and returns the unwrapped coordinates so that they may be used for analysis.
        """

        box_array = self.box_array  # Get the static box array --  NEED A NEW METHOD FOR V NEQ CONST SYSTEMS E.G NPT

        def Center_Box(positions_matrix):
            """ Center atoms in box

            A function to center the coordinates in the box so that the unwrapping is performed equally in all
            directions.

            args:
                positions_matrix (array) -- array of positions to be unwrapped.
            """

            positions_matrix -= (box_array[0] / 2)

        def Unwrap(database):
            """ Unwrap the coordinates

            Central function in the unwrapping method. This function will detect jumps across the box boundary and
            shifts the position of the atoms accordingly.
            """

            print("\n --- Beginning to unwrap coordinates --- \n")

            for item in self.species:
                # Construct the positions matrix -- Only temporary, we should make this memory safe
                positions_matrix = np.dstack((database[item]["Positions"]['x'],
                                              database[item]["Positions"]['y'],
                                              database[item]["Positions"]['z']))

                Center_Box(positions_matrix)  # Center the box at (0, 0, 0)

                for j in range(len(positions_matrix)):
                    difference = np.diff(positions_matrix[j], axis=0)  # Difference between all atoms in the array

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
                        positions_matrix[j][:, 0][box_cross[0][k]:] -= np.sign(difference[box_cross[0][k] - 1][0]) * \
                                                                       box_array[0]
                    for k in range(len(box_cross[1])):
                        positions_matrix[j][:, 1][box_cross[1][k]:] -= np.sign(difference[box_cross[1][k] - 1][1]) * \
                                                                       box_array[1]
                    for k in range(len(box_cross[2])):
                        positions_matrix[j][:, 2][box_cross[2][k]:] -= np.sign(difference[box_cross[2][k] - 1][2]) * \
                                                                       box_array[2]

                print(np.array([positions_matrix[i][:, 2] for i in range(len(positions_matrix))]))

                database[item].create_group("Unwrapped_Positions")
                database[item]["Unwrapped_Positions"].create_dataset('x',
                                                                     data=np.array([positions_matrix[i][:, 0] for i
                                                                                    in range(len(positions_matrix))]))
                database[item]["Unwrapped_Positions"].create_dataset('y',
                                                                     data=np.array([positions_matrix[i][:, 1] for i
                                                                                    in range(len(positions_matrix))]))
                database[item]["Unwrapped_Positions"].create_dataset('z',
                                                                     data=np.array([positions_matrix[i][:, 2] for i
                                                                                    in range(len(positions_matrix))]))

        with hf.File("{0}/{1}/{1}.hdf5".format(self.filepath, self.analysis_name), "r+") as database:
            Unwrap(database)

        print("\n --- Finished unwrapping coordinates --- \n")

    def Load_Matrix(self, identifier):
        """ Load a desired property matrix

        args:
            identifier (str) -- Name of the matrix to be loaded, e.g. Unwrapped_Positions, Velocities

        returns:
            Matrix of the property
        """

        property_matrix = [] # Define an empty list for the properties to fill

        with hf.File("{0}/{1}/{1}.hdf5".format(self.filepath, self.analysis_name), "r") as database:
            for item in self.species:
                property_matrix.append(np.dstack((database[item][identifier]['x'],
                                                   database[item][identifier]['y'],
                                                   database[item][identifier]['z'])))

        return property_matrix

    def Einstein_Diffusion_Coefficients(self):
        """ Calculate the Einstein self diffusion coefficients

            A function to implement the Einstein method for the calculation of the self diffusion coefficients
            of a liquid. In this method, unwrapped trajectories are read in and the MSD of the positions calculated and
            a gradient w.r.t time is calculated over several ranges to calculate an error measure.

            Data is loaded from the working directory.
        """

        # Load the matrix of species positions
        positions_matrix = self.Load_Matrix("Unwrapped_Positions")

        def Singular_Diffusion_Coefficients():
            """ Calculate singular diffusion coefficients

            Implement the Einstein method for the calculation of the singular diffusion coefficient. This is performed
            using unwrapped coordinates, generated by the unwrap method. From these values, the mean square displacement
            of each atom is calculated and averaged over all the atoms in the system.
            """

            diffusion_coefficients = {} # Define an empty dictionary to store the coefficients

            # Loop over each atomic specie to calculate self-diffusion
            for i in range(len(list(self.species))):
                msd = [[], [], []]
                for j in range(len(positions_matrix[i])):  # Loop over number of atoms of species i
                    for k in range(3):
                        msd[k].append((positions_matrix[i][j][:, k] - positions_matrix[i][j][0][k]) ** 2)

                msd = (1e-20)*np.sum([np.mean(msd[j], axis=0) for j in range(3)]) # Calculate the summed average of MSD

                time = np.linspace(self.time_dimensions[0], self.time_dimensions[1], len(msd))

                popt, pcov = curve_fit(Meta_Functions.Linear_Fitting_Function, time, msd)
                diffusion_coefficients[list(self.species)[i]] = popt[0] / 6

            return diffusion_coefficients

        def Distinct_Diffusion_Coefficients():
            """ Calculate the Distinct Diffusion Coefficients

            Use the Einstein method to calculate the distinct diffusion coefficients of the system from the mean
            square displacement, as calculated from different atoms. This value is averaged over all the possible
            combinations of atoms for the best fit.
            """
            pass

        singular_diffusion_coefficients = Singular_Diffusion_Coefficients()
        # Distinct_Diffusion_Coefficients()

        print("Einstein Self-Diffusion Coefficients: {0}".format(singular_diffusion_coefficients))

    def Green_Kubo_Diffusion_Coefficients(self):
        """ Calculate the Green_Kubo Diffusion coefficients

        Function to implement a Green-Kubo method for the calculation of diffusion coefficients whereby the velocity
        autocorrelation function is integrated over and divided by 3. Autocorrelation is performed using the scipy
        fft correlate function in order to speed up the calculation.
        """

        # load the velocity matrix
        velocity_matrix = self.Load_Matrix("Velocities")

        def Singular_Diffusion_Coefficients():
            """ Calculate the singular diffusion coefficients """

            diffusion_coefficients = {} # define empty dictionary for diffusion coefficients

            # Define time array - CHANGE to an input (you don't need all the time)
            time = np.linspace(self.time_dimensions[0], self.time_dimensions[1], len(velocity_matrix[0][0]))

            numerator = 1e-20
            denominator = 3*1e-24*(2*len(time)-1)
            prefactor = numerator/denominator

            # Loop over the species in the system
            for i in range(len(list(self.species.keys()))):
                vacf = np.zeros(len(velocity_matrix[i][0])) # Define vacf array

                # Loop over the atoms of species i to get the average
                for j in range(len(velocity_matrix[i])):
                    vacf += np.array(
                        signal.correlate(velocity_matrix[i][j][:, 0], velocity_matrix[i][j][:, 0], mode='same',
                                         method='fft') +
                        signal.correlate(velocity_matrix[i][j][:, 1], velocity_matrix[i][j][:, 1], mode='same',
                                         method='fft') +
                        signal.correlate(velocity_matrix[i][j][:, 2], velocity_matrix[i][j][:, 2], mode='same',
                                         method='fft'))

                # Calculate the self-diffusion coefficient for the species with the update prefactor
                prefactor /= len(velocity_matrix[i])
                diffusion_coefficients[list(self.species)[i]] = prefactor * np.trapz(vacf[int(len(vacf) / 2):], x=time)

            return diffusion_coefficients

        def Distinct_Diffusion_Coefficients():
            """ Calculate the distinct diffusion coefficients """

            diffusion_coefficients = {} # define empty dictionary for the coefficients

            species = list(self.species.keys())
            combinations = ['-'.join(tup) for tup in list(itertools.combinations_with_replacement(species, 2))]
            index_list = [i for i in range(len(velocity_matrix))]
            time = np.linspace(self.time_dimensions[0], self.time_dimensions[1], len(velocity_matrix[0][0]))

            # Update the dictionary with relevent combinations
            for item in combinations:
                diffusion_coefficients[item] = {}
            pairs = 0
            for tups in itertools.combinations_with_replacement(index_list, 2):

                # Define the multiplicative factor
                numerator = self.number_of_atoms*1e-20
                denominator = len(velocity_matrix[tups[0]])*len(velocity_matrix[tups[1]])*3*1e-24
                prefactor = numerator/denominator

                diff_array = []

                # Loop over reference atoms
                for i in range(len(velocity_matrix[tups[0]])):
                    # Loop over test atoms
                    vacf = np.zeros(len(velocity_matrix[tups[0]][i])) # initialize the vacf array
                    for j in range(len(velocity_matrix[tups[1]])):
                        # Add conditional statement to avoid i=j and alpha=beta
                        if tups[0] == tups[1] and j == i:
                            continue

                        vacf += np.array(
                            signal.correlate(velocity_matrix[tups[0]][i][:, 0], velocity_matrix[tups[1]][j][:, 0],
                                             mode='same', method='fft') +
                            signal.correlate(velocity_matrix[tups[0]][i][:, 1], velocity_matrix[tups[1]][j][:, 1],
                                             mode='same', method='fft') +
                            signal.correlate(velocity_matrix[tups[0]][i][:, 2], velocity_matrix[tups[1]][j][:, 2],
                                             mode='same', method='fft'))

                    diff_array.append(np.trapz(vacf, x=time))

                diffusion_coefficients[combinations[pairs]] = [prefactor*np.mean(diff_array),
                                                               prefactor*np.std(diff_array)/np.sqrt(len(diff_array))]
                pairs += 1
            return diffusion_coefficients

        #singular_diffusion_coefficients = Singular_Diffusion_Coefficients()
        a = Distinct_Diffusion_Coefficients()
        print(a)

        #for item in singular_diffusion_coefficients:
        #    print("Self-Diffusion Coefficient for {0} at {1}K: {2} m^2/s".format(item,
        #                                                                         self.temperature,
        #                                                                         singular_diffusion_coefficients[item]))

    def Nernst_Einstein_Conductivity(self, diffusion_coefficients, label):
        """ Calculate Nernst-Einstein Conductivity

        A function to determine the Nernst-Einstein as well as the corrected Nernst-Einstein
        conductivity of a system.

        args:
            diffusion_coefficients (dict) -- A dictionary of self and distinct diffusion coefficients
            label (str) -- A lable to identify whether the coefficients are from GK or Einstein methods

        returns:
            ionic_conductivity (dict) -- A dictionary of the ionic conductivity, corrected and uncorrected, with errors
        """
        pass

    def Einstein_Helfand_Conductivity(self, measurement_range):
        """ Calculate the Einstein-Helfand Conductivity

        A function to use the mean square displacement of the dipole moment of a system to extract the
        ionic conductivity

        args:
            measurement_range (int) -- time range over which the measurement should be performed
        """

        position_matrix = self.Load_Matrix("Unwrapped_Positions")
        summed_positions = [[] for i in range(len(list(self.species)))]

        # Sum up positions and calculate the dipole moment
        for i in range(len(position_matrix[0][0])):
            for j in range(len(summed_positions)):
                summed_positions[j].append(np.sum(position_matrix[j][:, i], axis=0))

        dipole_moment = (np.array(summed_positions[0]) - np.array(summed_positions[1]))
        dipole_moment_msd = [[], [], []] # Initialize empty dipole moment msd matrix

        loop_range = len(position_matrix[0][0]) - (measurement_range - 1) # Define the loop range

        # Fill the dipole moment msd matrix
        for i in range(loop_range):
            for j in range(3):
                dipole_moment_msd[j] += (dipole_moment[i:i + measurement_range, j] - dipole_moment[i][j])**2

        dipole_msd = np.array(np.sum(dipole_moment_msd)) # Calculate the net msd

        time = np.linspace(0.0, 15, len(dipole_msd)) # Initialize the time

        sigma_array = [] # Initialize and array for the conductivity calculations

        # Loop over different fit ranges to generate an array of conductivities, from which a mean and error can be
        # calculated
        for i in range(1000):
            # Create the measurment range
            start = np.random.randint(int(0.2*len(dipole_msd)), int(len(dipole_msd) - 2000))
            stop = np.random.randint(int(start + 1000), int(len(dipole_msd)))
            
            # Calculate the value and append the array
            popt, pcov = curve_fit(func, time[start:stop], dipole_msd[start:stop])
            sigma_array.append(popt[0])

        # Define the multiplicative prefactor of the calculation
        denominator = (6 * self.temperature * (self.volume * 1E-30) * Constants.boltzmann_constant)*(1e-9)*loop_range
        numerator = (1e-20)*(Constants.elementary_charge**2)
        prefactor = numerator/denominator

        sigma = prefactor*np.mean(sigma_array)
        sigma_error = prefactor*np.sqrt(np.var(sigma_array))

        print("Einstein-Helfand Conductivity at {0}K: {1} +- {2} S/cm^2".format(self.temperature,
                                                                                sigma / 100,
                                                                                sigma_error / 100))

    def Green_Kubo_Conductivity(self, data_range, plot=False):
        """ Calculate Green-Kubo Conductivity

        A function to use the current autocorrelation function to calculate the Green-Kubo ionic conductivity of the
        system being studied.

        args:
            data_range (int) -- number of data points with which to calculate the conductivity
            plot (bool) -- Decision to plot the normalized acf
        """

        velocity_matrix = self.Load_Matrix("Velocities")

        summed_velocity = [] # Define array for the summed velocities
        jacf = np.zeros(data_range) # Define the empty JACF array
        time = np.linspace(0, self.sample_rate*self.time_step*data_range, int(0.5*len(jacf)))  # define the time

        if plot == True:
            averaged_jacf = np.zeros(int(0.5*data_range))

        for i in range(len(list(self.species))):
            summed_velocity.append(np.sum(velocity_matrix[i][:, 0:], axis=0))

        current = (np.array(summed_velocity[0]) - np.array(summed_velocity[1])) # We need to fix these things
        loop_range = len(current) - data_range - 1 # Define the loop range
        sigma = []
        for i in range(loop_range):
            jacf = np.zeros(data_range)  # Define the empty JACF array
            jacf += (signal.correlate(current[:, 0][i:i + data_range],
                                      current[:, 0][i:i + data_range],
                                      mode='same', method='fft') +
                    signal.correlate(current[:, 1][i:i + data_range],
                                     current[:, 1][i:i + data_range],
                                     mode='same', method='fft') +
                    signal.correlate(current[:, 2][i:i + data_range],
                                     current[:, 2][i:i + data_range],
                                     mode='same', method='fft'))

            # Cut off the second half of the acf
            jacf = jacf[int((len(jacf) / 2)):]
            if plot == True:
                averaged_jacf += jacf

            numerator = (Constants.elementary_charge**2)*(1e-20)
            denominator = 3*Constants.boltzmann_constant*self.temperature*(self.volume*(1e-30))*(1e-12)*\
                          (2 * len(jacf) - 1)*2
            prefactor = numerator / denominator

            sigma.append(prefactor * np.trapz(jacf, x=time))

        if plot == True:
            averaged_jacf /= max(averaged_jacf)
            Methods.Trajectory_Methods.Plot_Investigation(self, [time, averaged_jacf], observable='Green_Kubo_Conductivity')

        print("Green-Kubo Ionic Conductivity at {0}K: {1} +- {2} S/cm^2".format(self.temperature, np.mean(sigma) / 100,
                                                                                (np.std(sigma)/(np.sqrt(len(sigma))))/100))
        return np.mean(sigma)

    def Green_Kubo_Viscosity(self):
        """ Calculate the shear viscosity of the system using Green Kubo

        Use a Green Kubo relation to calculate the shear viscosity of the system. This involves the calculation
        of the autocorrelation for the stress tensor of the sysetem.
        """
        pass

    def Radial_Distribution_Function(self, bins=1000, cutoff=None):
        """ Calculate the radial distribtion function

        This function will calculate the radial distribution function for all pairs available in the system.

        kwargs:
            bins (int) -- Number of bins to use in the histogram when building the distribution function
        """

        # Define cutoff to half a box vector if none other is specified
        if cutoff == None:
            cutoff = self.box_array[0] / 2

        positions_matrix = self.Load_Matrix("Positions") # Load the positions
        bin_width = cutoff / bins # Calculate the bin_width

    def Kirkwood_Buff_Integrals(self):
        """ Calculate the Kirkwood-Buff integrals for the system

        Function to calculate all possible kirkwood buff integrals in the trajectory data
        """
        pass

    def Structure_Factor(self):
        """ Calculate the structure factors in the system

        Function to calculate the possible structure factors for the system
        """
        pass

    def Angular_Distribution_Function(self):
        """ Calculate angular distribution functions

        Function to caluclate the possible angular distribution functions for the system
        """

