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


class TrajectoryThermal(Methods.Trajectory_Methods):
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
        self.Thermal_Conductivity = {"Einstein-Helfand": {},
                                   "Green-Kubo": {}}

        if self.new_project == False:
            self.Load_Class()
        else:
            self.Build_Database()



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

    def Get_LAMMPS_flux_file(self):
        """ Flux files are usually dumped with the fix print in lammps. Any global property can be printed there,
        important one for this case are the flux resulting from the compute

        :return:
        """

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
                for i in tqdm(range(int(self.number_of_configurations / self.batch_size))):
                    test = Methods.Trajectory_Methods.Read_Configurations(self, self.batch_size, f)

                    Methods.Trajectory_Methods.Process_Configurations(self, test, database, counter)

                    counter += self.batch_size

        self.Save_Class()

        print("\n ** Database has been constructed and saved for {0} ** \n".format(self.analysis_name))


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

    def Einstein_Helfand_Conductivity(self, measurement_range, plot=False, species=None):
        """ Calculate the Einstein-Helfand Conductivity

        A function to use the mean square displacement of the dipole moment of a system to extract the
        ionic conductivity

        args:
            measurement_range (int) -- time range over which the measurement should be performed
            plot(bool=False) -- If True, will plot the MSD over time
        """
        print("Sorry, this functionality is currently unavailable - check back in soon!")
        return


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

        current = (np.array(summed_velocity[0]) - np.array(summed_velocity[1]))  # We need to fix these things
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

