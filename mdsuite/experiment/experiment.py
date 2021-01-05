"""
Authors: Samuel Tovey, Francisco Torres
Affiliation: Institute for Computational Physics, University of Stuttgart ; 
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Purpose: Class functionality of the program
"""
import json
import os
import sys

import h5py as hf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import warnings
from pathlib import Path
import tensorflow as tf

import mdsuite.utils.constants as constants
import mdsuite.experiment.experiment_methods as methods

# File readers
from mdsuite.file_io.lammps_trajectory_files import LAMMPSTrajectoryFile

# Analysis modules
from mdsuite.analysis import einstein_diffusion_coefficients
from mdsuite.analysis import green_kubo_diffusion_coefficients
from mdsuite.analysis import green_kubo_ionic_conductivity
from mdsuite.analysis import einstein_helfand_ionic_conductivity
from mdsuite.analysis import radial_distribution_function
from mdsuite.analysis import coordination_number_calculation
from mdsuite.analysis import potential_of_mean_force
from mdsuite.analysis import kirkwood_buff_integrals
from mdsuite.analysis import structure_factor

from mdsuite.analysis.computations_dict import dict_classes_computations

# Transformation modules
from mdsuite.transformations import unwrap_coordinates

plt.style.use('bmh')
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class Experiment(methods.ProjectMethods):
    """ Experiment from simulation

    ...

    Attributes
    -----------

        trajectory_file : str
                            A file containing trajectory data of a simulation

        analysis_name : str
                            The name of the analysis being performed e.g. NaCl_1400K

        storage_path : str
                            Path to where the data should be stored (best to have  drive capable of storing large files)

        temperature : float
                            The temperature of the simulation that should be used in some analysis

        time_step : float
                            Time step of the simulation e.g 0.002

        volume : float
                            Volume of the simulation box

        species : dict
                            A dictionary of the species in the system and their properties. Their properties includes
                            index location in the trajectory file, mass of the species as taken from the PubChem
                            database, and the charge taken from the same database. When using these properties, it is
                            best that users confirm this information, with exception to the indices as they are read
                            from the file and will be correct.

        number_of_atoms : int
                            The total number of atoms in the simulation

        properties : dict
                            Properties in the trajectory available for analysis, not important for understanding

        property_groups : dict
                            Property groups, e.g Forces, Positions, Velocities, Torques,  along with their
                            location in the trajectory file.

        dimensions : float
                            Dimensionality of the system e.g. 3.0. This is currently not used anywhere in a useful way.
                            It is called in the calculations of some properties, but these properties cannot really be
                            calculated in any dimension other than 3 at the moment. Therefore this attribute is here
                            mostly for future functionality.

        box_array : list
                            Box lengths of the simulation, e.g [13.1, 22, 8.0]. It should be noted that at the moment
                            only cuboid structures can be used. If a non-rectangular box is parsed, the code will
                            read it in as a cuboid.

        number_of_configurations : int
                            The number of configurations in the trajectory

        units : dict
                            A dictionary of the to-SI unit conversion depending on the units used during the simulation.
                            In this code we stick to LAMMPS units conventions.

        diffusion_coefficients : dict
                            A dictionary of diffusion coefficients including from Einstein and Green-Kubo,
                            and split again into singular and distinct coefficients.

        ionic_conductivity : dict
                            Ionic conductivity of the system given by several different calculations including the
                            Green-Kubo approach, the Einstein-Helfand approach, the Nernst-Einstein, and the Corrected
                            Nernst-Einstein approaches.

        thermal_conductivity : dict
                            The thermal conductivity of the material. Can be calculated from a flux file or from local
                            atomic energies. These different values are stored as key: value pairs in the dictionary.

        Methods
        -------

        add_data(self, trajectory_file=None)
                            Add simulation data to the experiment for further analysis.

        unwrap_coordinates(self, species=None, center_box=True)
                            Unwrap the coordinates in the hdf5 database.

        load_matrix(self, identifier, species=None)
                            Load a data matrix from the hdf5 database. Not a private method as it is called by the
                            analysis modules which, whilst using experiment class methods and attributes, are not
                            by definition child classes and do not formally inherit.

        einstein_diffusion_coefficients(self, plot=False, singular=True, distinct=False, species=None, data_range=500)
                            Calculate the diffusion coefficients of the system constituents using the Einstein method.

        green_kubo_diffusion_coefficients(self, data_range=500, plot=False, singular=True, distinct=False, species=None)
                            Calculate the diffusion coefficients of the system constituents using the Green-Kubo method.

        nernst_einstein_conductivity(self)
                            Calculate the ionic conductivity of the system using the Nernst-Einstein and corrected
                            Nernst-Einstein methods.

        einstein_helfand_ionic_conductivity(self, data_range, plot=False)
                            Calculate the ionic conductivity of the system using the Einstein-Helfand method.

        green_kubo_ionic_conductivity(self, data_range, plot=False)
                            Calculate the ionic conductivity of the system using the Green-Kubo method.

        radial_distribution_function(self, plot=True, bins=500, cutoff=None, data_range=500)
                            Calculate the radial distribution function for the particle pairs in the system.

        calculate_coordination_numbers(self, plot=True)
                            Calculate the coordination numbers from the radial distribution functions.
    """

    def __init__(self, analysis_name, storage_path='./', timestep=1.0, temperature=0, units='real'):
        """ Initialise with trajectory_file """

        # Taken upon instantiation
        self.analysis_name = analysis_name
        self.storage_path = storage_path
        self.temperature = temperature
        self.time_step = timestep

        # Added from trajectory file
        self.trajectory_file = None
        self.sample_rate = None
        self.batch_size = None
        self.volume = None
        self.species = None
        self.number_of_atoms = None
        self.properties = None
        self.property_groups = None
        self.dimensions = None
        self.box_array = None
        self.number_of_configurations = 0
        self.time_dimensions = None
        self.units = self.units_to_si(units)

        # Properties of the experiment
        # TODO: maybe we could put all of this in a single structure.
        self.diffusion_coefficients = {"Einstein": {"Singular": {}, "Distinct": {}},
                                       "Green-Kubo": {"Singular": {}, "Distinct": {}}}
        self.ionic_conductivity = {"Einstein-Helfand": {},
                                   "Green-Kubo": {},
                                   "Nernst-Einstein": {"Einstein": None, "Green-Kubo": None},
                                   "Corrected Nernst-Einstein": {"Einstein": None, "Green-Kubo": None}}
        self.thermal_conductivity = {'Global': {"Green-Kubo": {}}}
        self.coordination_numbers = {}
        self.potential_of_mean_force_values = {}
        self.radial_distribution_function_state = False  # Set true if this has been calculated
        self.kirkwood_buff_integral_state=True  # Set true if it has been calculated

        self.results = {
            'diffusion_coefficients': self.diffusion_coefficients,
            'ionic_conductivity': self.ionic_conductivity,
            'thermal_conductivity': self.thermal_conductivity,
            'coordination_numbers': self.coordination_numbers,
            'potential_of_mean_force_values': self.potential_of_mean_force_values,
            'radial_distribution_function': self.radial_distribution_function_state,
            'kirkwood_buff_integral': self.kirkwood_buff_integral_state
        }


        test_dir = Path(f"{self.storage_path}/{self.analysis_name}")
        if test_dir.exists():
            print("This experiment already exists! I'll load it up now.")
            self._load_class()
        else:
            print("Creating a new experiment! How exciting!")
            self._build_model()

    def _process_input_file(self):
        """ Process the input file

        A trivial function to get the format of the input file. Will probably become more useful when we add support
        for more file formats.
        """

        if self.trajectory_file[-6:] == 'extxyz':
            file_format = 'extxyz'
        else:
            file_format = 'lammps_traj'

        return file_format

    def _get_system_properties(self):
        """ Get the properties of the system

        This method will call the Get_X_Properties depending on the file format. This function will update all of the
        class attributes and is necessary for the operation of the Build database method.

        args:
            file_format (str) -- Format of the file being read
        """

        file_format = self._process_input_file()  # Collect file format information
        trajectory_reader = self._select_file_reader(file_format)

        return trajectory_reader

    def _select_file_reader(self, argument):
        """ Switcher function to select relevant file reader """

        switcher = {
            'lammps_traj': LAMMPSTrajectoryFile
        }

        choice = switcher.get(argument, lambda: "Invalid filetype")

        return choice(self)

    def _build_model(self):
        """ Build the 'experiment' for the analysis

        A method to build the database in the hdf5 format. Within this method, several other are called to develop the
        database skeleton, get configurations, and process and store the configurations. The method is accompanied
        by a loading bar which should be customized to make it more interesting.
        """

        # Create new analysis directory and change into it
        try:
            os.mkdir(f'{self.storage_path}/{self.analysis_name}')  # Make the experiment directory
            os.mkdir(f'{self.storage_path}/{self.analysis_name}/Figures')  # Create a directory to save images
            os.mkdir(f'{self.storage_path}/{self.analysis_name}/data')  # Create a directory for data

        except FileExistsError:
            return

        self._save_class()

        print(f"** An experiment has been added entitled {self.analysis_name} **")

    def _get_minimal_class_state(self):
        """ Get a minimum umber of class properties for comparison """

        return [self.number_of_atoms, list(self.species), self.box_array]

    def _update_database(self):
        """ Update a pre-existing database """

        trajectory_reader = self._get_system_properties()  # select the correct trajectory reader
        # get properties of new trajectory
        compare_data = trajectory_reader.process_trajectory_file(update_class=False)
        class_state = self._get_minimal_class_state()

        if compare_data[:-1] == class_state:
            self.number_of_configurations += compare_data[3]

            trajectory_reader.resize_database()  # resize the database to accommodate the new data

            self._fill_database(trajectory_reader, counter=int(self.number_of_configurations - compare_data[3]))
        else:
            print(compare_data[:-1] == class_state)

    def _fill_database(self, trajectory_reader, counter=0):
        """ Loads data into a hdf5 database """

        loop_range = int((self.number_of_configurations - counter) / self.batch_size)
        with hf.File("{0}/{1}/{1}.hdf5".format(self.storage_path, self.analysis_name), "r+") as database:
            with open(self.trajectory_file) as f:
                for _ in tqdm(range(loop_range), ncols=70):
                    batch_data = trajectory_reader.read_configurations(self.batch_size, f)

                    trajectory_reader.process_configurations(batch_data, database, counter)

                    counter += self.batch_size

    def _build_new_database(self):
        """ Build a new database """

        trajectory_reader = self._get_system_properties()  # select the correct trajectory reader
        trajectory_reader.process_trajectory_file()  # get properties of the trajectory and update the class
        trajectory_reader.build_database_skeleton()  # Build the database skeleton

        self._fill_database(trajectory_reader)

        self.build_species_dictionary()  # Beef up the species dictionary
        self._save_class()

    def add_data(self, trajectory_file=None):
        """ Add data to the database """

        if trajectory_file is None:
            print("No data has been given")
            sys.exit()

        self.trajectory_file = trajectory_file  # Update the current class trajectory file

        # Check to see if a database exists
        test_db = Path(f"{self.storage_path}/{self.analysis_name}/{self.analysis_name}.hdf5")
        if test_db.exists():
            self._update_database()
        else:
            self._build_new_database()

        self._save_class()

    def unwrap_coordinates(self, species=None, center_box=True):
        """ unwrap coordinates of trajectory

        For a number of properties the input data must in the form of unwrapped coordinates. This function takes the
        stored trajectory and returns the unwrapped coordinates so that they may be used for analysis.
        """

        transformation_ufb = unwrap_coordinates.CoordinateUnwrapper(self, species, center_box)  # load the unwrapper
        transformation_ufb.unwrap_particles()  # unwrap the coordinates

    def load_matrix(self, identifier, species=None, select_slice=None, tensor=False):
        """ Load a desired property matrix

        args:
            identifier (str) -- Name of the matrix to be loaded, e.g. Unwrapped_Positions, Velocities
            species (list) -- List of species to be loaded

        returns:
            Matrix of the property
        """

        if species is None:
            species = list(self.species.keys())
        if select_slice is None:
            select_slice = np.s_[:]
        property_matrix = []  # Define an empty list for the properties to fill

        with hf.File(f"{self.storage_path}/{self.analysis_name}/{self.analysis_name}.hdf5", "r+") as database:
            for item in list(species):
                # Unwrap the positions if they need to be unwrapped
                if identifier == "Unwrapped_Positions" and "Unwrapped_Positions" not in database[item]:
                    print("We first have to unwrap the coordinates... Doing this now")
                    self.unwrap_coordinates(species=[item])
                if identifier not in database[item]:
                    print("This data was not found in the database. Was it included in your simulation input?")
                    return
                if tensor:
                    property_matrix.append(
                        tf.convert_to_tensor(np.dstack((database[item][identifier]['x'][select_slice],
                                                        database[item][identifier]['y'][select_slice],
                                                        database[item][identifier]['z'][select_slice])),
                                             dtype=tf.float64))
                else:
                    property_matrix.append(np.dstack((database[item][identifier]['x'][select_slice],
                                                      database[item][identifier]['y'][select_slice],
                                                      database[item][identifier]['z'][select_slice])))

        if len(property_matrix) == 1:
            return property_matrix[0]
        else:
            return property_matrix

    def einstein_diffusion_coefficients(self, plot=False, singular=True, distinct=False, species=None, data_range=500):
        """ Calculate the Einstein self diffusion coefficients

            A function to implement the Einstein method for the calculation of the self diffusion coefficients
            of a liquid. In this method, unwrapped trajectories are read in and the MSD of the positions calculated and
            a gradient w.r.t time is calculated over several ranges to calculate an error measure.

            args:
                plot (bool = False) -- If True, a plot of the msd will be displayed
                Singular (bool = True) -- If True, will calculate the singular diffusion coefficients
                Distinct (bool = False) -- If True, will calculate the distinct diffusion coefficients
                species (list) -- List of species to analyze
                data_range (int) -- Range over which the values should be calculated
        """

        if species is None:
            species = list(self.species.keys())

        calculation_ed = einstein_diffusion_coefficients.EinsteinDiffusionCoefficients(self, plot=plot,
                                                                                       singular=singular,
                                                                                       distinct=distinct,
                                                                                       species=species,
                                                                                       data_range=data_range)

        calculation_ed.run_analysis()

        self._save_class()  # Update class state

    def green_kubo_diffusion_coefficients(self, data_range=500, plot=False, singular=True, distinct=False,
                                          species=None):
        """ Calculate the Green_Kubo Diffusion coefficients

        Function to implement a Green-Kubo method for the calculation of diffusion coefficients whereby the velocity
        autocorrelation function is integrated over and divided by 3. Autocorrelation is performed using the scipy
        fft correlate function in order to speed up the calculation.
        """

        # Load all the species if none are specified
        if species is None:
            species = list(self.species.keys())

        calculation_gkd = green_kubo_diffusion_coefficients.GreenKuboDiffusionCoefficients(self, plot=plot,
                                                                                           singular=singular,
                                                                                           distinct=distinct,
                                                                                           species=species,
                                                                                           data_range=data_range)

        calculation_gkd.run_analysis()  # run the analysis

        self._save_class()  # Update class state

    def nernst_einstein_conductivity(self):
        """ Calculate Nernst-Einstein Conductivity

        A function to determine the Nernst-Einstein (NE) as well as the corrected Nernst-Einstein (CNE)
        conductivity of a system.

        TODO: (FRAN) I think this should not be here.
        """
        truth_array = [[bool(self.diffusion_coefficients["Einstein"]["Singular"]),
                        bool(self.diffusion_coefficients["Einstein"]["Distinct"])],
                       [bool(self.diffusion_coefficients["Green-Kubo"]["Singular"]),
                        bool(self.diffusion_coefficients["Green-Kubo"]["Distinct"])]]

        def _ne_conductivity(_diffusion_coefficients):
            """ Calculate the standard Nernst-Einstein Conductivity for the system

            args:
                _diffusion_coefficients (dict) -- dictionary of diffusion coefficients
            """

            numerator = self.number_of_atoms * (constants.elementary_charge ** 2)
            denominator = constants.boltzmann_constant * self.temperature * (self.volume * (self.units['length'] ** 3))
            prefactor = numerator / denominator

            diffusion_array = []
            for element in self.species:
                diffusion_array.append(_diffusion_coefficients["Singular"][element] *
                                       abs(self.species[element]['charge'][0]) *
                                       (len(self.species[element]['indices']) / self.number_of_atoms))

            return (prefactor * np.sum(diffusion_array)) / 100

        def _cne_conductivity(_singular_diffusion_coefficients, _distinct_diffusion_coefficients):
            print("Sorry, this currently isn't available")
            return

            numerator = self.number_of_atoms * (constants.elementary_charge ** 2)
            denominator = constants.boltzmann_constant * self.temperature * (self.volume * (self.units['length'] ** 3))
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

            self.ionic_conductivity["Nernst-Einstein"]["Einstein"] = _ne_conductivity(
                self.diffusion_coefficients["Einstein"])
            self.ionic_conductivity["Nernst-Einstein"]["Green-Kubo"] = _ne_conductivity(
                self.diffusion_coefficients["Green-Kubo"])

            print(f'Nernst-Einstein Conductivity from Einstein Diffusion: '
                  f'{self.ionic_conductivity["Nernst-Einstein"]["Einstein"]} S/cm\n'
                  f'Nernst-Einstein Conductivity from Green-Kubo Diffusion: '
                  f'{self.ionic_conductivity["Nernst-Einstein"]["Green-Kubo"]} S/cm')

        elif truth_array[0][0] is True and not any(truth_array[1]) is True:
            """ Calc just NE for EIN """

            self.ionic_conductivity["Nernst-Einstein"]["Einstein"] = _ne_conductivity(
                self.diffusion_coefficients["Einstein"])
            print(f'Nernst-Einstein Conductivity from Einstein Diffusion: '
                  f'{self.ionic_conductivity["Nernst-Einstein"]["Einstein"]} S/cm')

        elif truth_array[1][0] is True and not any(truth_array[0]) is True:
            """ Calc just NE for GK """

            self.ionic_conductivity["Nernst-Einstein"]["Green-Kubo"] = _ne_conductivity(
                self.diffusion_coefficients["Green-Kubo"])
            print(f'Nernst-Einstein Conductivity from Green-Kubo Diffusion: '
                  f'{self.ionic_conductivity["Nernst-Einstein"]["Green-Kubo"]} S/cm')

        elif all(truth_array[0]) is True and truth_array[1][0] is True:
            """ Calc CNE for EIN and just NE for GK"""
            pass

        elif all(truth_array[1]) is True and truth_array[0][0] is True:
            """ Calc CNE for GK and just NE for EIN"""
            pass

        else:
            print("This really should not be possible... something has gone horrifically wrong")
            return

        self._save_class()  # Update class state

    def einstein_helfand_ionic_conductivity(self, data_range=500, plot=True):
        """ Calculate the Einstein-Helfand Ionic Conductivity

        A function to use the mean square displacement of the dipole moment of a system to extract the
        ionic conductivity

        Parameters
        ----------
        data_range : int
                            Time range over which the measurement should be performed
        plot : bool
                            If True, will plot the MSD over time
        """

        calculation_ehic = einstein_helfand_ionic_conductivity.EinsteinHelfandIonicConductivity(self,
                                                                                                data_range=data_range,
                                                                                                plot=plot)
        calculation_ehic.run_analysis()
        self._save_class()

    def green_kubo_ionic_conductivity(self, data_range, plot=True):
        """ Calculate Green-Kubo Ionic Conductivity

        A function to use the current autocorrelation function to calculate the Green-Kubo ionic conductivity of the
        system being studied.

        Parameters
        ----------
        data_range : int
                            Number of data points with which to calculate the conductivity

        plot : bool
                            If True, a plot of the current autocorrelation function will be generated

        Returns
        -------
        sigma : float
                            The ionic conductivity in units of S/cm

        """

        calculation_gkic = green_kubo_ionic_conductivity.GreenKuboIonicConductivity(self, plot=plot,
                                                                                    data_range=data_range)
        calculation_gkic.run_analysis()

        self._save_class()  # Update class state

    def radial_distribution_function(self, plot=True, bins=500, cutoff=None):
        """ Calculate the radial distribution function """

        calculation_rdf = radial_distribution_function.RadialDistributionFunction(self, plot=plot,
                                                                                  bins=bins,
                                                                                  cutoff=cutoff)
        calculation_rdf.run_analysis()  # run the analysis
        self.radial_distribution_function_state = True  # update the analysis state
        self._save_class()  # save the class state

    def calculate_coordination_numbers(self, plot=True):
        """ Calculate the coordination numbers """

        calculation_cn = coordination_number_calculation.CoordinationNumbers(self, plot)
        calculation_cn.run_analysis()
        self._save_class()

    def potential_of_mean_force(self, plot=True, save=True):
        """ Calculate the potential of mean-force """

        calculation_pomf = potential_of_mean_force.PotentialOfMeanForce(self, plot=plot, save=save)
        calculation_pomf.run_analysis()

    def kirkwood_buff_integrals(self, plot=True, save=True):
        """ Calculate the kirkwood buff integrals """

        calculation_kbi = kirkwood_buff_integrals.KirkwoodBuffIntegral(self, plot=plot, save=save)
        calculation_kbi.run_analysis()
        self.kirkwood_buff_integral_state = True  # set the value to true for future

    # TODO def green_kubo_viscosity(self):

    # TODO def structure_factor(self):
    def structure_factor(self):
        calculation_strfac = structure_factor.StructureFactor(self)
        calculation_strfac.run_analysis()

    # TODO def angular_distribution_function(self):

    def run_computation(self, computation_name, **kwargs):
        """ Run a computation

        The type of computation will be stored in a dictionary.

        Parameters
        ----------
        computation_name : str
                            name of the computation to be performed

        **kwargs : extra arguments passed to the classes


        Returns
        -------
        sigma : float
                            The ionic conductivity in units of S/cm

        """

        print(dict_classes_computations)
        print(computation_name)
        try:
            class_compute = dict_classes_computations[computation_name]
        except KeyError:
            # TODO: maybe this exception can be done better, but I dont know enough about handling exceptions.
            print(f'{computation_name} not found')
            print(f'Available computations are:')
            [print(key) for key in dict_classes_computations.keys()]
            sys.exit(1)

        object_compute = class_compute(self, **kwargs)
        object_compute.run_analysis()
        self._save_class()

    @staticmethod
    def help_computations_args(computation_name):
        """
        Shows the input parameters for the specified class
        """
        try:
            class_compute = dict_classes_computations[computation_name]
        except KeyError:
            # TODO: maybe this exception can be done better, but I dont know enough about handling exceptions.
            print(f'{computation_name} not found')
            print(f'Available computations are:')
            [print(key) for key in dict_classes_computations.keys()]
            sys.exit(1)
        print(help(class_compute))

    def dump_results_json(self):
        filename = Path(f"{self.storage_path}/{self.analysis_name}.json")
        with open(filename, 'w') as fp:
            json.dump(self.results, fp, indent=4, sort_keys=True)