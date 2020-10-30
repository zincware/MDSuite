"""
Author: Samuel Tovey ; Francisco Torres
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Purpose: Class functionality of the program
"""

import os

import h5py as hf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from tqdm import tqdm
import warnings

import mdsuite.constants as Constants
import mdsuite.meta_functions as Meta_Functions
import mdsuite.methods as Methods

plt.style.use('bmh')
warnings.filterwarnings("ignore")
tqdm.monitor_interval = 0


class ProjectThermal(Methods.ProjectMethods):
    """ Experiment from simulation

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

        number_of_atoms (int) -- number of atoms in the system

        properties (dict) -- properties in the trajectory available for analysis, not important for understanding

        dimensions (float) -- dimensionality of the system e.g. 3.0

        box_array (list) -- box lengths, e.g [10.7, 10.7, 10.8]

        number_of_configurations (int) -- number of configurations in the trajectory (timesteps)

        time_dimensions (list) -- Time domain in the system, e.g, for a 1ns simulation, [0.0, 1e-9]

        thermal_conductivity (float) -- Thermal conductivity of the system e.g. 4.5 W/m/K
    """

    def __init__(self, analysis_name, new_project=False, storage_path=None,
                 temperature=None, time_step=None, time_unit=None, filename=None, length_unit=None,
                 number_of_atoms=None, volume=None):
        """ Initialise with filename """

        self.filename = filename
        self.analysis_name = analysis_name
        self.new_project = new_project
        self.filepath = storage_path
        self.temperature = temperature
        self.time_step = time_step
        self.time_unit = time_unit
        self.length_unit = length_unit
        self.number_of_atoms = number_of_atoms
        self.volume = volume
        self.time_0 = 0
        self.sample_rate = None
        self.batch_size = None
        self.species = None
        self.properties = None
        self.dimensions = None
        self.box_array = None
        self.number_of_configurations = None
        self.number_of_blocks = None
        self.time_dimensions = None
        self.n_lines_header = None
        self.thermal_conductivity = {"Green-Kubo": {}}

        if not self.new_project:
            self.load_class()
        else:
            self.build_database()

    def process_input_file(self):
        """ Process the input file

        A trivial function to get the format of the input file. Will probably become more useful when we add support
        for more file formats.
        """
        filename, file_extension = os.path.splitext(self.filename)
        file_format = file_extension[1:]  # remove the dot.

        return file_format

    def get_system_properties(self, file_format):
        """ Get the properties of the system

        This method will call the get_X_groperties depending on the file format. This function will update all of the
        class attributes and is necessary for the operation of the Build database method.

        args:
            file_format (str) -- Format of the file being read
        """
        supported_file_formats = {
            'dat': self.get_lammps_flux_file,
        }

        # if the format is listed in the dict above, it will run
        # otherwise, give not implemented error
        supported_file_formats.get(file_format, self.not_implemented)()

    def get_lammps_flux_file(self):
        """ Flux files are usually dumped with the fix print in lammps. Any global property can be printed there,
        important one for this case are the flux resulting from the compute

        """
        properties_summary = {}
        lammps_properties_labels = {'time', 'temp', 'c_flux[1]',
                                    'c_flux[2]', 'c_flux[3]'}

        self.n_lines_header = 0  # number of lines of header
        with open(self.filename) as f:
            header = []
            for line in f:
                self.n_lines_header += 1
                if line.startswith("#"):
                    header.append(line.split())
                else:
                    varibles_lammps = line.split()  # after the comments, we have the line with the variables
                    break

        with open(self.filename) as f:
            number_of_configurations = sum(1 for _ in f) - self.n_lines_header

        # Find properties available for analysis
        for position, variable in enumerate(varibles_lammps):
            if variable in lammps_properties_labels:
                properties_summary[variable] = position

        batch_size = Meta_Functions.optimize_batch_size(self.filename, number_of_configurations)

        # get time related properties of the system
        with open(self.filename) as f:
            # skip the header
            for _ in range(self.n_lines_header):
                next(f)
            time_0_line = f.readline().split()
            time_0 = float(time_0_line[properties_summary['time']])
            time_1_line = f.readline().split()
            time_1 = float(time_1_line[properties_summary['time']])

        sample_rate = (time_1 - time_0) / self.time_step
        time_n = (number_of_configurations - number_of_configurations % batch_size) * sample_rate

        # Update class attributes with calculated data
        self.batch_size = batch_size
        self.properties = properties_summary
        self.number_of_configurations = number_of_configurations
        self.time_dimensions = [0.0, time_n * self.time_step * self.time_unit]
        self.sample_rate = sample_rate
        self.time_0 = time_0

        # Get the number of atoms if not set in initialization
        if self.number_of_atoms is None:
            self.number_of_atoms = int(header[2][1])  # hopefully always in the same position
        # Get the volume, if not set in initialization
        if self.volume is None:
            self.volume = float(header[4][7])  # hopefully always in the same position

    def build_database(self):
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
        self._build_database_skeleton()

        print("Beginning Build database")

        self.number_of_blocks = int(self.number_of_configurations / self.batch_size)

        with hf.File("{0}/{1}/{1}.hdf5".format(self.filepath, self.analysis_name), "r+") as database:
            with open(self.filename) as f:
                # Skip header lines (this type of file has only header at the beginning)
                for j in range(self.n_lines_header):
                    f.readline()

                # start the counter
                counter = 0
                for _ in tqdm(range(self.number_of_blocks)):
                    test = self.read_configurations(self.batch_size, f)

                    self.process_configurations(test, database, counter)

                    counter += self.batch_size

        self.save_class()

        print("\n ** Database has been constructed and saved for {0} ** \n".format(self.analysis_name))


    def _build_database_skeleton(self):
        database = hf.File('{0}/{1}/{1}.hdf5'.format(self.filepath, self.analysis_name), 'w', libver='latest')

        # Build the database structure
        for property_in in self.properties:
            database.create_dataset(property_in, (self.number_of_configurations -
                                                  self.number_of_configurations % self.batch_size,),
                                    compression="gzip", compression_opts=9)

    def process_configurations(self, data, database, counter):
        for lammps_var, column_num in self.properties.items():
            # grab the corresponding column and set it as numbers
            column_data = data[:, column_num].astype(float)

            # remove the time offset
            if lammps_var == 'time':
                column_data = (column_data - self.time_0) * self.time_unit

            # LAMMPS uses a weird unit for the flux being in energy*velocity units
            # This is done so that the user can then divide by the appropriate volume.
            # The volume is considered in the method Green_Kubo_Conductivity_Thermal
            # This is the required change for Real Units
            kcal2j = 4186.0 / Constants.avogadro_constant

            if 'c_flux' in lammps_var:
                column_data = column_data * kcal2j * self.length_unit / self.time_unit

            # copy it to the database
            database[lammps_var][counter: counter + self.batch_size] = column_data

    @staticmethod
    def read_configurations(n_configurations, f):
        """ Read in N configurations

        This function will read in N configurations from the file that has been opened previously by the parent method.

        args:

            n_configurations (int) -- Number of configurations to read in. This will depend on memory availability and the size of each
                        configuration. Automatic setting of this variable is not yet available and therefore, it will be set
                        manually.
            f (obj) --
        """
        data = []

        for i in range(n_configurations):
            data.append(f.readline().split())

        return np.array(data)

    def load_column(self, identifier):
        """ Load a desired column from the hdf5

        args:
            identifier (str) -- Name of the matrix to be loaded, e.g. Unwrapped_Positions, Velocities
            species (list) -- List of species to be loaded

        returns:
            Matrix of the property
        """

        with hf.File(f"{self.filepath}/{self.analysis_name}/{self.analysis_name}.hdf5", "r+") as database:
            if identifier not in database:
                print("This data was not found in the database. Was it included in your simulation input?")
                return

            column_data = np.array(database[identifier])

        return column_data

    def load_flux_matrix(self):
        """ Load the flux matrix

        returns:
            Matrix of the property flux
        """
        identifiers = [f'c_flux[{i + 1}]' for i in range(3)]
        matrix_data = []

        for identifier in identifiers:
            column_data = self.load_column(identifier)
            matrix_data.append(column_data)
        matrix_data = np.array(matrix_data).T  # transpose such that [timestep, dimension]
        return matrix_data

    def green_kubo_conductivity_thermal(self, data_range, plot=False):
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

        fluxes = self.load_flux_matrix()

        time = np.linspace(0, self.sample_rate * self.time_step * data_range * self.time_unit,
                           data_range)  # define the time

        if plot == True:
            averaged_jacf = np.zeros(data_range)

        # prepare the prefactor for the integral
        numerator = 1
        denominator = 3 * (data_range / 2 - 1) * self.temperature ** 2 * Constants.boltzmann_constant \
                      * self.volume * self.length_unit ** 3
        # not sure why I need the /2 in data range...
        prefactor = numerator / denominator

        loop_range = len(fluxes) - data_range - 1  # Define the loop range
        sigma = []

        # main loop for computation
        for i in tqdm(range(loop_range)):
            jacf = np.zeros(2 * data_range - 1)  # Define the empty JACF array
            jacf += (signal.correlate(fluxes[:, 0][i:i + data_range],
                                      fluxes[:, 0][i:i + data_range],
                                      mode='full', method='fft') +
                     signal.correlate(fluxes[:, 1][i:i + data_range],
                                      fluxes[:, 1][i:i + data_range],
                                      mode='full', method='fft') +
                     signal.correlate(fluxes[:, 2][i:i + data_range],
                                      fluxes[:, 2][i:i + data_range],
                                      mode='full', method='fft'))

            # Cut off the second half of the acf
            jacf = jacf[int((len(jacf) / 2)):]
            if plot:
                averaged_jacf += jacf

            integral = np.trapz(jacf, x=time)
            sigma.append(integral)

        sigma = prefactor * np.array(sigma)

        if plot:
            averaged_jacf /= max(averaged_jacf)
            plt.plot(time, averaged_jacf)
            plt.xlabel("Time (s)")
            plt.ylabel("Normalized Current Autocorrelation Function")
            plt.savefig(f"GK_Cond_{self.temperature}.pdf", )
            plt.show()

        print(f"Green-Kubo Ionic Conductivity at {self.temperature}K: {np.mean(sigma)} +- "
              f"{np.std(sigma) / np.sqrt(len(sigma))} W/m/K")

        self.save_class()  # Update class state

    def not_implemented(self):
        raise NotImplementedError
