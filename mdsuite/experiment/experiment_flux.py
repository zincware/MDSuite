"""
Summary
-------
"""

import os
import warnings

import h5py as hf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

plt.style.use('bmh')
warnings.filterwarnings("ignore")
tqdm.monitor_interval = 0

import mdsuite.utils.meta_functions as meta_functions
import mdsuite.experiment.experiment_methods as methods
from mdsuite.calculators.flux_analyses import _GreenKuboThermalConductivityFlux

class ProjectFlux(methods.ProjectMethods):
    """
    Experiment from simulation

    Attributes:

        filename (str) -- trajectory_file of the trajectory

        analysis_name (str) -- name of the analysis being performed e.g. NaCl_1400K

        new_project (bool) -- If the project has already been build, if so, the class state will be loaded

        storage_path (str) -- where to store the data (best to have  drive capable of storing large files)

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
                 temperature=None, units=None, filename=None,
                 number_of_atoms=None, volume=None, time_step=None):
        """ Initialise with trajectory_file """

        self.filename = filename
        self.analysis_name = analysis_name
        self.new_project = new_project
        self.storage_path = storage_path
        self.temperature = temperature
        self.number_of_atoms = number_of_atoms
        self.time_step = time_step
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
        self.units = self.units_to_si(units)

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
        lammps_properties_labels = {'time', 'temp', 'c_flux'}  # words to be searched in the file

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
            # Find which variables are provided. Accepts partial matches: c_flux_thermal[1] will be accepted for example
            # new words can be added in the set lammps_properties_labels
            if any(word in variable for word in lammps_properties_labels):
                properties_summary[variable] = position

        batch_size = meta_functions.optimize_batch_size(self.filename, number_of_configurations)

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
        self.time_dimensions = [0.0, time_n * self.time_step * self.units['time']]
        self.sample_rate = sample_rate
        self.time_0 = time_0

        # Get the number of atoms if not set in initialization
        if self.number_of_atoms is None:
            self.number_of_atoms = int(header[2][1])  # hopefully always in the same position

        # Get the volume, if not set in initialization
        if self.volume is None:
            self.volume = float(header[4][7])  # hopefully always in the same position

    def build_database(self):
        """
        Build the 'database' for the analysis

        A method to build the database in the hdf5 format. Within this method, several other are called to develop the
        database skeleton, get configurations, and process and store the configurations. The method is accompanied
        by a loading bar which should be customized to make it more interesting.
        """

        # Create new analysis directory and change into it
        try:
            os.mkdir('{0}/{1}'.format(self.storage_path, self.analysis_name))
        except FileExistsError:
            pass

        file_format = self.process_input_file()  # Collect data array
        self.get_system_properties(file_format)  # Update class attributes
        self._build_database_skeleton()

        print("Beginning Build database")

        self.number_of_blocks = int(self.number_of_configurations / self.batch_size)

        with hf.File("{0}/{1}/{1}.hdf5".format(self.storage_path, self.analysis_name), "r+") as database:
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

        print(f"\n ** Database has been constructed and saved for {self.analysis_name}  ** \n")

    def _build_database_skeleton(self):
        """
        We need to override the method because the flux files have a different structure
        """
        database = hf.File('{0}/{1}/{1}.hdf5'.format(self.storage_path, self.analysis_name), 'w', libver='latest')

        # Build the database structure
        for property_in in self.properties:
            database.create_dataset(property_in, (self.number_of_configurations -
                                                  self.number_of_configurations % self.batch_size,),
                                    compression="gzip", compression_opts=9)

    def process_configurations(self, data, database, counter):
        """
        Processes the input data and converts units if needed.
        TODO: add conversions for new types (viscosity, etc)

        Parameters
        ----------
        data
        database
        counter
        """

        def _convert_time(time_array):
            # removes the time offset
            return (time_array - self.time_0) # * self.units['time']
        #
        # def _convert_heat_units(c_flux_thermal_array):
        #     # adjustes units for heatflux
        #     return c_flux_thermal_array * self.units['energy'] * self.units['length'] / self.units['time']
        #
        # conversions_dict = {"c_flux_thermal": _convert_heat_units,
        #                     "time": _convert_time}
        conversions_dict = {"time": _convert_time}

        for lammps_var, column_num in self.properties.items():
            # grab the corresponding column and set it as numbers
            column_data = data[:, column_num].astype(float)

            try:
                conversion_keyword = [word for word in conversions_dict.keys() if word in lammps_var][0]
            except IndexError:
                conversion_keyword = None

            if conversion_keyword:
                column_data = conversions_dict[conversion_keyword](column_data)

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

        with hf.File(f"{self.storage_path}/{self.analysis_name}/{self.analysis_name}.hdf5", "r+") as database:
            if identifier not in database:
                print("This data was not found in the database. Was it included in your simulation input?")
                return

            column_data = np.array(database[identifier])

        return column_data

    def not_implemented(self):
        raise NotImplementedError

    def green_kubo_thermal_conductivity(self, data_range, plot=False):
        """ Calculate the thermal conductivity using Green-Kubo formalism

        args:
            data_range (int) -- time range over which the measurement should be performed
        kwargs:
        """
        print(data_range)
        calculation_ehic = _GreenKuboThermalConductivityFlux(self, data_range=data_range, plot=plot)
        calculation_ehic._compute_thermal_conductivity()
        self.save_class()