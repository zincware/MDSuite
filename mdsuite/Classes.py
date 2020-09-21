"""
Author: Samuel Tovey
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Purpose: Class functionality of the program
"""

import numpy as np
import os
import sys
from scipy import signal
from scipy.optimize import curve_fit
import mdsuite.Methods as Methods
import pickle
import h5py as hf
import mdsuite.Constants as Constants
import mdsuite.Meta_Functions as Meta_Functions
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.style.use('bmh')


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

        length_unit (float) -- scaling factor for the lengths, should results in SI units (m), e.g. 1e-10 for angstroms

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
                 temperature=None, time_step=None, time_unit=None, filename=None, length_unit=None):
        """ Initialise with filename """

        self.filename = filename
        self.analysis_name = analysis_name
        self.new_project = new_project
        self.filepath = storage_path
        self.temperature = temperature
        self.time_step = time_step
        self.time_unit = time_unit
        self.length_unit = length_unit
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
        self.Diffusion_Coefficients = {"Einstein": {"Singular": {}, "Distinct": {}},
                                       "Green-Kubo": {"Singular": {}, "Distinct": {}}}
        self.Ionic_Conductivity = {"Einstein-Helfand": {},
                                   "Green-Kubo": {},
                                   "Nernst-Einstein": {"Einstein": None, "Green-Kubo": None},
                                   "Corrected Nernst-Einstein": {"Einstein": None, "Green-Kubo": None}}

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

        attributes = []
        for item in vars(self).items():
            attributes.append(item)
        for tuple in attributes:
            print(f"{tuple[0]}: {tuple[1]}")

        return attributes

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
            self.Get_LAMMPS_Properties()
        else:
            self.Get_EXTXYZ_Properties()

    def Build_Database(self):
        """ Build the 'database' for the analysis

        A method to build the database in the hdf5 format. Within this method, several other are called to develop the
        database skeleton, get configurations, and process and store the configurations. The method is accompanied
        by a loading bar which should be customized to make it more interesting.
        """

        # Create new analysis directory and change into it
        try:
            os.mkdir('{0}/{1}'.format(self.filepath, self.analysis_name))
        except FileExistsError:
            pass

        file_format = self.Process_Input_File()  # Collect data array
        self.Get_System_Properties(file_format)  # Update class attributes
        self.Build_Database_Skeleton()

        print("Beginning Build database")

        with hf.File("{0}/{1}/{1}.hdf5".format(self.filepath, self.analysis_name), "r+") as database:
            with open(self.filename) as f:
                counter = 0
                for i in tqdm(range(int(self.number_of_configurations / self.batch_size))):
                    test = self.Read_Configurations(self.batch_size, f)

                    self.Process_Configurations(test, database, counter)

                    counter += self.batch_size

        self.Build_Species_Dictionary() # Beef up the species dictionary
        self.Save_Class()

        print("\n ** Database has been constructed and saved for {0} ** \n".format(self.analysis_name))

    def Unwrap_Coordinates(self, species=None):
        """ Unwrap coordinates of trajectory

        For a number of properties the input data must in the form of unwrapped coordinates. This function takes the
        stored trajectory and returns the unwrapped coordinates so that they may be used for analysis.
        """

        box_array = self.box_array  # Get the static box array --  NEED A NEW METHOD FOR V NEQ CONST SYSTEMS E.G NPT

        if species == None:
            species = list(self.species.keys())

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

            for item in species:
                # Construct the positions matrix -- Only temporary, we should make this memory safe
                positions_matrix = self.Load_Matrix("Positions", [item])
                Center_Box(positions_matrix[0])  # Center the box at (0, 0, 0)

                for j in range(len(positions_matrix[0])):
                    difference = np.diff(positions_matrix[0][j], axis=0)  # Difference between all atoms in the array

                    # Indices where the atoms jump in the original array
                    box_jump = [np.where(abs(difference[:, 0]) >= (box_array[0] / 2))[0],
                                np.where(abs(difference[:, 1]) >= (box_array[1] / 2))[0],
                                np.where(abs(difference[:, 2]) >= (box_array[2] / 2))[0]]

                    # Indices of first box cross
                    box_cross = [box_jump[0] + 1, box_jump[1] + 1, box_jump[2] + 1]

                    for k in range(len(box_cross[0])):
                        positions_matrix[0][j][:, 0][box_cross[0][k]:] -= np.sign(difference[box_cross[0][k] - 1][0]) * \
                                                                       box_array[0]
                    for k in range(len(box_cross[1])):
                        positions_matrix[0][j][:, 1][box_cross[1][k]:] -= np.sign(difference[box_cross[1][k] - 1][1]) * \
                                                                       box_array[1]
                    for k in range(len(box_cross[2])):
                        positions_matrix[0][j][:, 2][box_cross[2][k]:] -= np.sign(difference[box_cross[2][k] - 1][2]) * \
                                                                       box_array[2]

                database[item].create_group("Unwrapped_Positions")
                database[item]["Unwrapped_Positions"].create_dataset('x',
                                                                     data=np.array([positions_matrix[0][i][:, 0] for i
                                                                                    in range(len(positions_matrix[0]))]))
                database[item]["Unwrapped_Positions"].create_dataset('y',
                                                                     data=np.array([positions_matrix[0][i][:, 1] for i
                                                                                    in range(len(positions_matrix[0]))]))
                database[item]["Unwrapped_Positions"].create_dataset('z',
                                                                     data=np.array([positions_matrix[0][i][:, 2] for i
                                                                                    in range(len(positions_matrix[0]))]))

        with hf.File("{0}/{1}/{1}.hdf5".format(self.filepath, self.analysis_name), "r+") as database:
            for item in species:
                if "Unwrapped_Positions" in database[item].keys():
                    print(f"{item} already has an unwrapped positions dataset, we will use that instead")
                    return
            Unwrap(database)
            print("\n --- Finished unwrapping coordinates --- \n")

    def Load_Matrix(self, identifier, species=None):
        """ Load a desired property matrix

        args:
            identifier (str) -- Name of the matrix to be loaded, e.g. Unwrapped_Positions, Velocities
            species (list) -- List of species to be loaded

        returns:
            Matrix of the property
        """

        if species == None:
            species = list(self.species.keys())
        property_matrix = []  # Define an empty list for the properties to fill

        with hf.File(f"{self.filepath}/{self.analysis_name}/{self.analysis_name}.hdf5", "r+") as database:
            for item in list(species):
                # Unwrap the positions if they need to be unwrapped
                if identifier == "Unwrapped_Positions" and "Unwrapped_Positions" not in database[item]:
                    print("We first have to unwrap the coordinates... Doing this now")
                    self.Unwrap_Coordinates(species=[item])
                if identifier not in database[item]:
                    print("This data was not found in the database. Was it included in your simulation input?")
                    return

                property_matrix.append(np.dstack((database[item][identifier]['x'],
                                                  database[item][identifier]['y'],
                                                  database[item][identifier]['z'])))

        return property_matrix

    def Einstein_Diffusion_Coefficients(self, plot=False, singular=True, distinct=False, species=None):
        """ Calculate the Einstein self diffusion coefficients

            A function to implement the Einstein method for the calculation of the self diffusion coefficients
            of a liquid. In this method, unwrapped trajectories are read in and the MSD of the positions calculated and
            a gradient w.r.t time is calculated over several ranges to calculate an error measure.

            args:
                plot (bool = False) -- If True, a plot of the msd will be displayed
                Singular (bool = True) -- If True, will calculate the singular diffusion coefficients
                Distinct (bool = False) -- If True, will calculate the distinct diffusion coefficients
                species (list) -- List of species to analyze
        """

        if species == None:
            species = list(self.species.keys())

        def Singular_Diffusion_Coefficients():
            """ Calculate singular diffusion coefficients

            Implement the Einstein method for the calculation of the singular diffusion coefficient. This is performed
            using unwrapped coordinates, generated by the unwrap method. From these values, the mean square displacement
            of each atom is calculated and averaged over all the atoms in the system.
            """

            used_configurations = int(self.number_of_configurations - self.number_of_configurations % self.batch_size)
            # Loop over each atomic specie to calculate self-diffusion
            for item in list(species):
                positions_matrix = self.Load_Matrix("Unwrapped_Positions", [item])
                msd = [[np.zeros(used_configurations)],
                       [np.zeros(used_configurations)],
                       [np.zeros(used_configurations)]]

                numerator = self.length_unit ** 2
                denominator = (len(self.species[item]['indices'])) * 6
                prefactor = numerator / denominator
                for j in tqdm(range(len(self.species[item]['indices']))):  # Loop over number of atoms of species i
                    for k in range(3):
                        msd[k] += (positions_matrix[0][j][:, k] -
                                   positions_matrix[0][j][0][k]) ** 2

                # Calculate the summed average of MSD
                msd = prefactor * (msd[0][0] + msd[1][0] + msd[2][0])
                time = np.linspace(self.time_dimensions[0], self.time_dimensions[1], len(msd))

                if plot == True:
                    plt.plot(time, msd, label=item)

                popt, pcov = curve_fit(Meta_Functions.Linear_Fitting_Function, time, msd)
                self.Diffusion_Coefficients["Einstein"]["Singular"][item] = popt[0]

            if plot == True:
                plt.legend()
                plt.show()

        def Distinct_Diffusion_Coefficients():
            """ Calculate the Distinct Diffusion Coefficients

            Use the Einstein method to calculate the distinct diffusion coefficients of the system from the mean
            square displacement, as calculated from different atoms. This value is averaged over all the possible
            combinations of atoms for the best fit.
            """
            print("Sorry, distinct diffusion from Einstein is not yet available - Check back soon!")

        if singular == True:
            Singular_Diffusion_Coefficients()
            print("Einstein Self-Diffusion Coefficients:\n")
            for item in self.Diffusion_Coefficients["Einstein"]['Singular']:
                print(f"{item}: {self.Diffusion_Coefficients['Einstein']['Singular'][item]} m^2/s\n")

        if distinct == True:
            Distinct_Diffusion_Coefficients()

        self.Save_Class() # Update class state

    def Green_Kubo_Diffusion_Coefficients(self, data_range, plot=False, singular=True, distinct=False, species=None):
        """ Calculate the Green_Kubo Diffusion coefficients

        Function to implement a Green-Kubo method for the calculation of diffusion coefficients whereby the velocity
        autocorrelation function is integrated over and divided by 3. Autocorrelation is performed using the scipy
        fft correlate function in order to speed up the calculation.
        """

        # Load all the species if none are specified
        if species == None:
            species = list(self.species.keys())

        def Singular_Diffusion_Coefficients():
            """ Calculate the singular diffusion coefficients """

            # Define time array - CHANGE to an input (you don't need all the time)
            time = np.linspace(0.0, data_range * self.time_step * self.sample_rate, data_range)

            numerator = self.length_unit ** 2
            denominator = 3 * (self.time_unit) * (len(time) - 1)
            prefactor = numerator / denominator

            # Loop over the species in the system
            for item in species:
                velocity_matrix = self.Load_Matrix("Velocities", [item])

                loop_values = np.linspace(0.1 * self.number_of_configurations,
                                          self.number_of_configurations - data_range - 1,
                                          100, dtype=int)
                loop_range = len(loop_values)
                coefficient_array = []

                if plot == True:
                    parsed_vacf = np.zeros(int(data_range))

                for i in tqdm(loop_values):
                    vacf = np.zeros(int(2 * data_range - 1))  # Define vacf array
                    # Loop over the atoms of species to get the average
                    for j in range(len(self.species[item]['indices'])):
                        vacf += np.array(
                            signal.correlate(velocity_matrix[0][j][i:i + data_range, 0],
                                             velocity_matrix[0][j][i:i + data_range, 0],
                                             mode='full', method='fft') +
                            signal.correlate(velocity_matrix[0][j][i:i + data_range, 1],
                                             velocity_matrix[0][j][i:i + data_range, 1],
                                             mode='full', method='fft') +
                            signal.correlate(velocity_matrix[0][j][i:i + data_range, 2],
                                             velocity_matrix[0][j][i:i + data_range, 2],
                                             mode='full', method='fft'))

                    coefficient_array.append((prefactor / len(self.species[item]['indices'])) * np.trapz(vacf[int(len(vacf) / 2):],
                                                                                              x=time))

                    if plot == True:
                        parsed_vacf += vacf[int(len(vacf) / 2):]

                self.Diffusion_Coefficients["Green-Kubo"]["Singular"][item] = np.mean(coefficient_array)

                if plot == True:
                    plt.plot(time, (parsed_vacf / loop_range) / max(parsed_vacf), label=item)

            if plot == True:
                plt.legend()
                plt.show()

        def Distinct_Diffusion_Coefficients():
            """ Calculate the distinct diffusion coefficients """

            print("Please note, distinct diffusion coefficients are not currently accurate")

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
                numerator = self.number_of_atoms * (self.length_unit ** 2)
                denominator = len(velocity_matrix[tups[0]]) * len(velocity_matrix[tups[1]]) * 3 * (self.time_unit ** 2)
                prefactor = numerator / denominator

                diff_array = []

                # Loop over reference atoms
                for i in range(len(velocity_matrix[tups[0]])):
                    # Loop over test atoms
                    vacf = np.zeros(len(velocity_matrix[tups[0]][i]))  # initialize the vacf array
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

                self.Diffusion_Coefficients["Green-Kubo"]["Distinct"][combinations[pairs]] = \
                    [prefactor * np.mean(diff_array), prefactor * np.std(diff_array) / np.sqrt(len(diff_array))]

                pairs += 1

        if singular == True:
            Singular_Diffusion_Coefficients()
            for item in species:
                print(f"Self-Diffusion Coefficient for {item} at {self.temperature}K: "
                      f"{self.Diffusion_Coefficients['Green-Kubo']['Singular'][item]} m^2/s")

        if distinct == True:
            Distinct_Diffusion_Coefficients()
            for item in self.Diffusion_Coefficients["Green-Kubo"]["Distinct"]:
                print(f"Distinct-Diffusion Coefficient for {item} at {self.temperature}K: "
                      f"{self.Diffusion_Coefficients['Green-Kubo']['Distinct'][item]} m^2/s")

        self.Save_Class() # Update class state

    def Nernst_Einstein_Conductivity(self):
        """ Calculate Nernst-Einstein Conductivity

        A function to determine the Nernst-Einstein (NE) as well as the corrected Nernst-Einstein (CNE)
        conductivity of a system.

        """
        truth_array = [[bool(self.Diffusion_Coefficients["Einstein"]["Singular"]),
                        bool(self.Diffusion_Coefficients["Einstein"]["Distinct"])],
                       [bool(self.Diffusion_Coefficients["Green-Kubo"]["Singular"]),
                        bool(self.Diffusion_Coefficients["Green-Kubo"]["Distinct"])]]

        def _NE_Conductivity(_diffusion_coefficients):
            """ Calculate the standard Nernst-Einstein Conductivity for the system

            args:
                _diffusion_coefficients (dict) -- dictionary of diffusion coefficients
            """

            numerator = self.number_of_atoms * (Constants.elementary_charge ** 2)
            denominator = Constants.boltzmann_constant * self.temperature * (self.volume * (self.length_unit ** 3))
            prefactor = numerator / denominator

            diffusion_array = []
            for element in self.species:
                diffusion_array.append(_diffusion_coefficients["Singular"][element] *
                                       abs(self.species[element]['charge'][0]) *
                                       (len(self.species[element]['indices']) / self.number_of_atoms))

            return (prefactor * np.sum(diffusion_array)) / 100

        def _CNE_Conductivity(_singular_diffusion_coefficients, _distinct_diffusion_coefficients):
            print("Sorry, this currently isn't available")
            return

            numerator = self.number_of_atoms * (Constants.elementary_charge ** 2)
            denominator = Constants.boltzmann_constant * self.temperature * (self.volume * (self.length_unit ** 3))
            prefactor = numerator / denominator

            singular_diffusion_array = []
            for element in self.species:
                singular_diffusion_array.append(_singular_diffusion_coefficients[element] *
                                                (len(self.species[element]['indices']) / self.number_of_atoms))

        if all(truth_array[0]) is True and all(truth_array[1]) is True:
            "Update all NE and CNE cond"
            pass

        elif not any(truth_array[0]) is True and not any(truth_array[1]) is True:
            "Run the diffusion analysis and then calc. all"
            pass

        elif all(truth_array[0]) is True and not any(truth_array[1]) is True:
            """ Calc NE, CNE for Einstein """
            pass

        elif all(truth_array[1]) is True and not any(truth_array[0]) is True:
            """ Calc all NE, CNE for GK """
            pass

        elif truth_array[0][0] is True and truth_array[1][0] is True:
            """ Calc just NE for EIN and GK """

            self.Ionic_Conductivity["Nernst-Einstein"]["Einstein"] = _NE_Conductivity(
                self.Diffusion_Coefficients["Einstein"])
            self.Ionic_Conductivity["Nernst-Einstein"]["Green-Kubo"] = _NE_Conductivity(
                self.Diffusion_Coefficients["Green-Kubo"])

            print(f'Nernst-Einstein Conductivity from Einstein Diffusion: '
                  f'{self.Ionic_Conductivity["Nernst-Einstein"]["Einstein"]} S/cm\n'
                  f'Nernst-Einstein Conductivity from Green-Kubo Diffusion: '
                  f'{self.Ionic_Conductivity["Nernst-Einstein"]["Green-Kubo"]} S/cm')

        elif truth_array[0][0] is True and not any(truth_array[1]) is True:
            """ Calc just NE for EIN """

            self.Ionic_Conductivity["Nernst-Einstein"]["Einstein"] = _NE_Conductivity(
                self.Diffusion_Coefficients["Einstein"])
            print(f'Nernst-Einstein Conductivity from Einstein Diffusion: '
                  f'{self.Ionic_Conductivity["Nernst-Einstein"]["Einstein"]} S/cm')

        elif truth_array[1][0] is True and not any(truth_array[0]) is True:
            """ Calc just NE for GK """

            self.Ionic_Conductivity["Nernst-Einstein"]["Green-Kubo"] = _NE_Conductivity(
                self.Diffusion_Coefficients["Green-Kubo"])
            print(f'Nernst-Einstein Conductivity from Green-Kubo Diffusion: '
                  f'{self.Ionic_Conductivity["Nernst-Einstein"]["Green-Kubo"]} S/cm')

        elif all(truth_array[0]) is True and truth_array[1][0] is True:
            """ Calc CNE for EIN and just NE for GK"""
            pass

        elif all(truth_array[1]) is True and truth_array[0][0] is True:
            """ Calc CNE for GK and just NE for EIN"""
            pass

        else:
            print("This really should not be possible... something has gone horrifically wrong")
            return

        self.Save_Class() # Update class state

    def Einstein_Helfand_Conductivity(self, data_range, plot=False, species=None):
        """ Calculate the Einstein-Helfand Conductivity

        A function to use the mean square displacement of the dipole moment of a system to extract the
        ionic conductivity

        args:
            data_range (int) -- time range over which the measurement should be performed
        kwargs:
            plot(bool=False) -- If True, will plot the MSD over time
        """

        position_matrix = self.Load_Matrix("Unwrapped_Positions")
        summed_positions = [[] for i in range(len(list(self.species)))]

        # Sum up positions and calculate the dipole moment
        for i in range(len(position_matrix[0][0])):
            for j in range(len(summed_positions)):
                summed_positions[j].append(np.sum(position_matrix[j][:, i], axis=0))


        for i in range(len(summed_positions)):
            if i == 0:
                dipole_moment = np.array(summed_positions[i]) * (self.species[list(self.species)[i]]['charge'][0])
            else:
                dipole_moment += np.array(summed_positions[i]) * (self.species[list(self.species)[i]]['charge'][0])

        dipole_moment_msd = [[np.zeros(data_range)],
                             [np.zeros(data_range)],
                             [np.zeros(data_range)]]  # Initialize empty dipole moment msd matrix

        loop_range = len(position_matrix[0][0]) - (data_range - 1)  # Define the loop range

        # Fill the dipole moment msd matrix
        for i in tqdm(range(loop_range)):
            for j in range(3):
                dipole_moment_msd[j] += (dipole_moment[i:i + data_range, j] - dipole_moment[i][j]) ** 2

        dipole_msd = np.array(np.array(dipole_moment_msd[0]) +
                              np.array(dipole_moment_msd[1]) +
                              np.array(dipole_moment_msd[2]))  # Calculate the net msd

        # Initialize the time
        time = np.linspace(0.0, data_range * self.sample_rate * self.time_step, len(dipole_msd[0]))

        sigma_array = []  # Initialize and array for the conductivity calculations
        # Loop over different fit ranges to generate an array of conductivities, from which a value can be calculated
        for i in range(100):
            # Create the data range
            start = np.random.randint(int(0.1 * len(dipole_msd[0])), int(0.60 * len(dipole_msd[0])))
            stop = np.random.randint(int(1.4 * start), int(1.65 * start))

            # Calculate the value and append the array
            popt, pcov = curve_fit(Meta_Functions.Linear_Fitting_Function, time[start:stop], dipole_msd[0][start:stop])
            sigma_array.append(popt[0])

        # Define the multiplicative prefactor of the calculation
        denominator = (6 * self.temperature * (self.volume * (self.length_unit) ** 3) * Constants.boltzmann_constant) * \
                      (self.time_unit) * loop_range
        numerator = (self.length_unit ** 2) * (Constants.elementary_charge ** 2)
        prefactor = numerator / denominator

        sigma = prefactor * np.mean(sigma_array)
        sigma_error = prefactor * (np.sqrt(np.var(sigma_array)) / np.sqrt(len(sigma_array)))

        if plot == True:
            plt.plot(time, dipole_msd[0])
            plt.xlabel("Time")
            plt.ylabel("Dipole Mean Square Displacement")
            plt.savefig(f"EHCond_{self.temperature}.pdf", format='pdf', dpi=600)
            plt.show()

        print(f"Einstein-Helfand Conductivity at {self.temperature}K: {sigma / 100} +- {sigma_error / 100} S/cm")

        self.Save_Class() # Update class state

    def Green_Kubo_Conductivity(self, data_range, plot=False, species=None):
        """ Calculate Green-Kubo Conductivity

        A function to use the current autocorrelation function to calculate the Green-Kubo ionic conductivity of the
        system being studied.

        args:
            data_range (int) -- number of data points with which to calculate the conductivity

        kwargs:
            plot (bool=False) -- If True, a plot of the current autocorrelation function will be generated

        returns:
            sigma (float) -- The ionic conductivity in units of S/cm

        """

        velocity_matrix = self.Load_Matrix("Velocities")

        summed_velocity = []  # Define array for the summed velocities
        time = np.linspace(0, self.sample_rate * self.time_step * data_range, data_range)  # define the time

        if plot == True:
            averaged_jacf = np.zeros(data_range)

        for i in range(len(list(self.species))):
            summed_velocity.append(np.sum(velocity_matrix[i][:, 0:], axis=0))

        for i in range(len(summed_velocity)):
            if i == 0:
                current = np.array(summed_velocity[i]) * (self.species[list(self.species)[i]]['charge'][0])
            else:
                current += np.array(summed_velocity[i]) * (self.species[list(self.species)[i]]['charge'][0])

        loop_range = len(current) - data_range - 1  # Define the loop range
        sigma = []
        for i in tqdm(range(loop_range)):
            jacf = np.zeros(2 * data_range - 1)  # Define the empty JACF array
            jacf += (signal.correlate(current[:, 0][i:i + data_range],
                                      current[:, 0][i:i + data_range],
                                      mode='full', method='fft') +
                     signal.correlate(current[:, 1][i:i + data_range],
                                      current[:, 1][i:i + data_range],
                                      mode='full', method='fft') +
                     signal.correlate(current[:, 2][i:i + data_range],
                                      current[:, 2][i:i + data_range],
                                      mode='full', method='fft'))

            # Cut off the second half of the acf
            jacf = jacf[int((len(jacf) / 2)):]
            if plot == True:
                averaged_jacf += jacf

            numerator = 2 * (Constants.elementary_charge ** 2) * (self.length_unit ** 2)
            denominator = 3 * Constants.boltzmann_constant * self.temperature * (
                    self.volume * (self.length_unit ** 3)) * \
                          self.time_unit * (2 * len(jacf) - 1)
            prefactor = numerator / denominator

            sigma.append(prefactor * np.trapz(jacf, x=time))

        if plot == True:
            averaged_jacf /= max(averaged_jacf)
            plt.plot(time, averaged_jacf)
            plt.xlabel("Time (ps)")
            plt.ylabel("Normalized Current Autocorrelation Function")
            plt.savefig(f"GK_Cond_{self.temperature}.pdf", )
            plt.show()

        print(f"Green-Kubo Ionic Conductivity at {self.temperature}K: {np.mean(sigma) / 100} +- "
              f"{0.01 * np.std(sigma) / np.sqrt(len(sigma))} S/cm")

        self.Save_Class() # Update class state

    def Green_Kubo_Viscosity(self):
        """ Calculate the shear viscosity of the system using Green Kubo

        Use a Green Kubo relation to calculate the shear viscosity of the system. This involves the calculation
        of the autocorrelation for the stress tensor of the sysetem.
        """
        print("Sorry, this functionality is currently unavailable - check back in soon!")
        return

    def Radial_Distribution_Function(self, bins=1000, cutoff=None):
        """ Calculate the radial distribtion function

        This function will calculate the radial distribution function for all pairs available in the system.

        kwargs:
            bins (int) -- Number of bins to use in the histogram when building the distribution function
        """

        print("Sorry, this functionality is currently unavailable - check back in soon!")
        # Define cutoff to half a box vector if none other is specified
        if cutoff == None:
            cutoff = self.box_array[0] / 2

        positions_matrix = self.Load_Matrix("Positions")  # Load the positions
        bin_width = cutoff / bins  # Calculate the bin_width

    def Kirkwood_Buff_Integrals(self):
        """ Calculate the Kirkwood-Buff integrals for the system

        Function to calculate all possible kirkwood buff integrals in the trajectory data
        """
        print("Sorry, this functionality is currently unavailable - check back in soon!")
        return

    def Structure_Factor(self):
        """ Calculate the structure factors in the system

        Function to calculate the possible structure factors for the system
        """
        print("Sorry, this functionality is currently unavailable - check back in soon!")
        return

    def Angular_Distribution_Function(self):
        """ Calculate angular distribution functions

        Function to caluclate the possible angular distribution functions for the system
        """
        print("Sorry, this functionality is currently unavailable - check back in soon!")
        return
