"""
The central experiment class fundamental to all analysis.

Summary
-------
The experiment class is the main class involved in characterizing and analyzing a simulation.
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

import mdsuite.experiment.experiment_methods as methods

# File readers
from mdsuite.file_io.lammps_trajectory_files import LAMMPSTrajectoryFile

# Calculator modules
from mdsuite.calculators import einstein_diffusion_coefficients
from mdsuite.calculators import green_kubo_diffusion_coefficients
from mdsuite.calculators import green_kubo_ionic_conductivity
from mdsuite.calculators import einstein_helfand_ionic_conductivity
from mdsuite.calculators import radial_distribution_function
from mdsuite.calculators import coordination_number_calculation
from mdsuite.calculators import potential_of_mean_force
from mdsuite.calculators import kirkwood_buff_integrals
from mdsuite.calculators import structure_factor

from mdsuite.calculators.computations_dict import dict_classes_computations

# Transformation modules
from mdsuite.transformations import unwrap_coordinates

plt.style.use('bmh')
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class Experiment(methods.ProjectMethods):
    """
    Experiment from simulation

    The central experiment class fundamental to all analysis.

    Attributes
    ----------

    trajectory_file : str
            A file containing trajectory data of a simulation

    analysis_name : str
            The name of the analysis being performed e.g. NaCl_1400K

    storage_path : str
            Path to where the data should be stored (best to have  drive capable of storing large files)

    temperature : float
            The temperature of the simulation that should be used in some analysis. Necessary as it cannot be easily
            read in from the simulation data.

    time_step : float
            Time step of the simulation e.g 0.002. Necessary as it cannot be easily read in from the trajectory.

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

    """

    def __init__(self, analysis_name, storage_path='./', timestep=1.0, temperature=0, units='real'):
        """
        Initialise the experiment class.
        """

        # Taken upon instantiation
        self.analysis_name = analysis_name    # Name of the experiment.
        self.storage_path = storage_path      # Where to store the data - should have sufficient free space.
        self.temperature = temperature        # Temperature of the system.
        self.time_step = timestep             # Timestep chosen for the simulation.

        # Added from trajectory file
        self.trajectory_file = None           # Name of the trajectory file.
        self.sample_rate = None               # Rate at which configurations are dumped in the trajectory.
        self.batch_size = None                # Number of configurations in each batch.
        self.volume = None                    # Volume of the system.
        self.species = None                   # Species dictionary.
        self.number_of_atoms = None           # Number of atoms in the simulation.
        self.properties = None                # Properties measured in the simulation.
        self.property_groups = None           # Names of the properties measured in the simulation
        self.dimensions = None                # Dimensionality of the system.
        self.box_array = None                 # Box vectors.
        self.number_of_configurations = 0     # Number of configurations in the trajectory.
        self.time_dimensions = None           # Good question
        self.units = self.units_to_si(units)  # Units used during the simulation.

        # Memory properties
        self.memory_requirements = {}

        # Properties of the experiment
        # TODO: maybe we could put all of this in a single structure.
        # self.system_measurements = {}         #  Properties measured during the analysis.
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
        self.kirkwood_buff_integral_state = True         # Set true if it has been calculated

        # Dictionary of results
        self.results = {
            'diffusion_coefficients': self.diffusion_coefficients,
            'ionic_conductivity': self.ionic_conductivity,
            'thermal_conductivity': self.thermal_conductivity,
            'coordination_numbers': self.coordination_numbers,
            'potential_of_mean_force_values': self.potential_of_mean_force_values,
            'radial_distribution_function': self.radial_distribution_function_state,
            'kirkwood_buff_integral': self.kirkwood_buff_integral_state
        }

        test_dir = Path(f"{self.storage_path}/{self.analysis_name}")  # get the theoretical directory

        # Check if the experiment exists and load if it does.
        if test_dir.exists():
            print("This experiment already exists! I'll load it up now.")
            self.load_class()
        else:
            print("Creating a new experiment! How exciting!")
            self._build_model()

    def _process_input_file(self):
        """
        Process the input file

        A trivial function to get the format of the input file. Will probably become more useful when we add support
        for more file formats.

        Returns
        -------
        file_format : str
                Format of the input file.
        """
        # TODO: This needs to be completely reformatted to read in general file formats or take is as an argument.
        # Check for the file format.
        if self.trajectory_file[-6:] == 'extxyz':
            file_format = 'extxyz'
        else:
            file_format = 'lammps_traj'

        return file_format

    def _get_system_properties(self):
        """
        Get the properties of the system

        This method will call the Get_X_Properties depending on the file format. This function will update all of the
        class attributes and is necessary for the operation of the Build database method.

        Returns
        -------
        trajectory_class : object
                Instance of a file reader class associated with the file format.
        """

        file_format = self._process_input_file()                   # Collect file format information
        trajectory_reader = self._select_file_reader(file_format)  # select the trajectory reader

        return trajectory_reader

    def _select_file_reader(self, argument):
        """
        Switcher function to select relevant file reader.

        Parameters
        ----------
        argument : str
                Name of the trajectory format to be read in.

        Returns
        -------
        choice : object
                Returns the instance of the trajectory reader associated with the file format being read.
        """

        # Switcher argument for selecting class
        switcher = {
            'lammps_traj': LAMMPSTrajectoryFile
        }

        choice = switcher.get(argument, lambda: "Invalid filetype")  # get the trajectory reader.

        return choice(self)

    def _build_model(self):
        """
        Build the 'experiment' for the analysis

        A method to build the database in the hdf5 format. Within this method, several other are called to develop the
        database skeleton, get configurations, and process and store the configurations. The method is accompanied
        by a loading bar which should be customized to make it more interesting.
        """

        # Create new analysis directory and change into it
        try:
            os.mkdir(f'{self.storage_path}/{self.analysis_name}')          # Make the experiment directory
            os.mkdir(f'{self.storage_path}/{self.analysis_name}/Figures')  # Create a directory to save images
            os.mkdir(f'{self.storage_path}/{self.analysis_name}/data')     # Create a directory for data

        except FileExistsError:  # throw exception if the file exits
            return

        self.save_class()  # save the class state.

        print(f"** An experiment has been added entitled {self.analysis_name} **")

    def _get_minimal_class_state(self):
        """
        Get a minimum umber of class properties for comparison.

        Returns
        -------
        minimal class state : list
                Returns a minimal state of the class used to ensure that new trajectory data is of the same form as the
                preexisting entries.
        """

        return [self.number_of_atoms, list(self.species), self.box_array]

    def _update_database(self):
        """
        Update a pre-existing database with new trajectory data.
        """

        trajectory_reader = self._get_system_properties()  # select the correct trajectory reader

        compare_data = trajectory_reader.process_trajectory_file(update_class=False)  # get properties of new trajectory

        class_state = self._get_minimal_class_state()  # get properties of the preexisting data

        if compare_data[:-1] == class_state:
            self.number_of_configurations += compare_data[3]

            trajectory_reader.resize_database()  # resize the database to accommodate the new data

            # Add the new data to the database.
            self._fill_database(trajectory_reader, counter=int(self.number_of_configurations - compare_data[3]))

        else:
            print("Added data does not match the data in the database, make a new experiment.")

    def collect_memory_information(self):
        """
        Get information about dataset memory requirements

        This method will simply get the size of all the datasets in the database such that efficient memory management
        can be performed during analysis.
        """

        with hf.File("{0}/{1}/{1}.hdf5".format(self.storage_path, self.analysis_name), "r+") as db:
            for item in self.species:                               # Loop over the species keys
                self.memory_requirements[item] = {}                 # construct a new dict entry
                for group in db[item]:                              # Loop over property groups
                    memory = 0                                      # Dummy variable for memory
                    for dataset in db[item][group]:                 # Loop over the datasets in the group
                        memory += db[item][group][dataset].nbytes   # Sum over the memory for each dataset
                    self.memory_requirements[item][group] = memory  # Update the dictionary.

    def _fill_database(self, trajectory_reader, counter=0):
        """
        Loads data into a hdf5 database

        Parameters
        ----------
        trajectory_reader : object
                Instance of a trajectory reader class.

        counter : int
                Number of configurations that have been read in.
        """

        loop_range = int((self.number_of_configurations - counter) / self.batch_size)    # loop range for the data.
        with hf.File("{0}/{1}/{1}.hdf5".format(self.storage_path, self.analysis_name), "r+") as database:
            with open(self.trajectory_file) as f:
                for _ in tqdm(range(loop_range), ncols=70):
                    batch_data = trajectory_reader.read_configurations(self.batch_size, f)   # load the batch data
                    trajectory_reader.process_configurations(batch_data, database, counter)  # process the trajectory
                    counter += self.batch_size                                               # Update counter

    def _build_new_database(self):
        """
        Build a new database
        """
        trajectory_reader = self._get_system_properties()  # select the correct trajectory reader
        trajectory_reader.process_trajectory_file()        # get properties of the trajectory and update the class
        trajectory_reader.build_database_skeleton()        # Build the database skeleton
        self._fill_database(trajectory_reader)             # Fill the database with trajectory data
        self.build_species_dictionary()                    # Add data to the species dictionary.
        self.save_class()                                 # Update the class state

    def add_data(self, trajectory_file=None):
        """
        Add data to the database

        Parameters
        ----------
        trajectory_file : str
                Trajectory file to be process and added to the database.
        """

        # Check if there is a trajectory file.
        if trajectory_file is None:
            print("No data has been given")
            return  # exit method as nothing more can be done

        self.trajectory_file = trajectory_file  # Update the current class trajectory file

        # Check to see if a database exists
        test_db = Path(f"{self.storage_path}/{self.analysis_name}/{self.analysis_name}.hdf5")  # get theoretical path.
        if test_db.exists():
            self._update_database()
        else:
            self._build_new_database()

        self.collect_memory_information()      # Update the memory information
        self.save_class()                      # Update the class state.

    def unwrap_coordinates(self, species=None, center_box=True):
        """
        Unwrap coordinates of trajectory

        For a number of properties the input data must in the form of unwrapped coordinates. This function takes the
        stored trajectory and returns the unwrapped coordinates so that they may be used for analysis.

        Parameters
        ----------
        species : list
                Species on which to apply the transformation.
        center_box : bool
                Decision on whether or not to center the box data

        See Also
        --------
        mdsuite.transformations.unwrap_coordinates.CoordinateUnwrapper
        """

        transformation_ufb = unwrap_coordinates.CoordinateUnwrapper(self, species, center_box)  # load the transform.
        transformation_ufb.unwrap_particles()                                                   # unwrap the coordinates

    def load_matrix(self, identifier, species=None, select_slice=None, tensor=False, scalar=False, sym_matrix=False):
        """
        Load a desired property matrix.

        Parameters
        ----------
        identifier : str
                Name of the matrix to be loaded, e.g. Unwrapped_Positions, Velocities
        species : list
                List of species to be loaded
        select_slice : np.slice
                A slice to select from the database.
        tensor : bool
                If true, the data will be returned as a tensorflow tensor.
        scalar : bool
                If true, the data will be returned as a scalar array
        sym_matrix : bool
                If true, data will be returned as as stress tensor format.

        Returns
        -------
        property_matrix : np.array, tf.tensor
                Tensor of the property to be studied. Format depends on kwargs.
        """

        # If no species list is given, use all species in the Experiment class instance.
        if species is None:
            species = list(self.species.keys())  # get list of all species available.

        # If no slice is given, load all configurations.
        if select_slice is None:
            select_slice = np.s_[:]  # set the numpy slice object.

        property_matrix = []  # Define an empty list for the properties to fill

        with hf.File(f"{self.storage_path}/{self.analysis_name}/{self.analysis_name}.hdf5", "r+") as database:
            for item in list(species):

                # Unwrap the positions if they need to be unwrapped
                if identifier == "Unwrapped_Positions" and "Unwrapped_Positions" not in database[item]:
                    print("We first have to unwrap the coordinates... Doing this now")
                    self.unwrap_coordinates(species=[item])  # perform the coordinate unwrapping.

                # Check if the desired property is in the database.
                if identifier not in database[item]:
                    print("This data was not found in the database. Was it included in your simulation input?")
                    return

                # If the tensor kwarg is True, return a tensor.
                if tensor:
                    property_matrix.append(
                        tf.convert_to_tensor(np.dstack((database[item][identifier]['x'][select_slice],
                                                        database[item][identifier]['y'][select_slice],
                                                        database[item][identifier]['z'][select_slice])),
                                             dtype=tf.float64))

                elif sym_matrix:  # return a stress tensor
                    property_matrix.append(np.dstack((database[item][identifier]['x'][select_slice],
                                                      database[item][identifier]['y'][select_slice],
                                                      database[item][identifier]['z'][select_slice],
                                                      database[item][identifier]['xy'][select_slice],
                                                      database[item][identifier]['xz'][select_slice],
                                                      database[item][identifier]['yz'][select_slice],)))
                elif scalar:  # return a scalar
                    property_matrix.append(database[item][identifier][select_slice])

                else:  # return a numpy array
                    property_matrix.append(np.dstack((database[item][identifier]['x'][select_slice],
                                                      database[item][identifier]['y'][select_slice],
                                                      database[item][identifier]['z'][select_slice])))

        # Check if the property loaded was a scalar.
        if len(property_matrix) == 1:
            return property_matrix[0]  # return the scalar dataset
        else:
            return property_matrix     # return the full tensor object.

    def einstein_diffusion_coefficients(self, plot=False, singular=True, distinct=False, species=None, data_range=500):
        """
        Calculate the Einstein self diffusion coefficients

        A function to implement the Einstein method for the calculation of the self diffusion coefficients
        of a liquid. In this method, unwrapped trajectories are read in and the MSD of the positions calculated and
        a gradient w.r.t time is calculated over several ranges to calculate an error measure.

        Parameters
        ----------
        plot : bool
                If True, a plot of the msd will be displayed
        singular : bool
                If True, will calculate the singular diffusion coefficients
        distinct : bool
                If True, will calculate the distinct diffusion coefficients
        species : list
                List of species to analyze
        data_range : int
                Range over which the values should be calculated
        """

        # If no species are given, use them all.
        if species is None:
            species = list(self.species.keys())  # generate list of available species.

        # Instantiate the diffusion class.
        calculation_ed = einstein_diffusion_coefficients.EinsteinDiffusionCoefficients(self, plot=plot,
                                                                                       singular=singular,
                                                                                       distinct=distinct,
                                                                                       species=species,
                                                                                       data_range=data_range)
        calculation_ed.run_analysis()  # perform the calculation
        self.save_class()             # Update class state

    def green_kubo_diffusion_coefficients(self, data_range=500, plot=False, singular=True, distinct=False, species=None):
        """
        Calculate the Green_Kubo Diffusion coefficients

        Function to implement a Green-Kubo method for the calculation of diffusion coefficients whereby the velocity
        autocorrelation function is integrated over and divided by 3. Autocorrelation is performed using the scipy
        fft correlate function in order to speed up the calculation.

        Parameters
        ----------
        data_range : int
                number of time steps to be used in the analysis.
        plot : bool
                Decision on whether or not to plot the data.
        singular : bool
                If true, singular diffusion coefficients will be calculated.
        distinct : bool
                If true, the distinct diffusion coefficients will be calculated.
        species : list
                Species for which the calculation should be performed.
        """

        # Load all the species if none are specified
        if species is None:
            species = list(self.species.keys())

        # Instantiate the class
        calculation_gkd = green_kubo_diffusion_coefficients.GreenKuboDiffusionCoefficients(self, plot=plot,
                                                                                           singular=singular,
                                                                                           distinct=distinct,
                                                                                           species=species,
                                                                                           data_range=data_range)
        calculation_gkd.run_analysis()  # run the analysis
        self.save_class()              # Update class state

    def nernst_einstein_conductivity(self):
        """
        Calculate Nernst-Einstein Conductivity

        A function to determine the Nernst-Einstein (NE) as well as the corrected Nernst-Einstein (CNE)
        conductivity of a system.
        """
        # TODO: Write the NE conductivity class
        self.save_class()  # Update class state

    def einstein_helfand_ionic_conductivity(self, data_range=500, plot=True):
        """
        Calculate the Einstein-Helfand Ionic Conductivity

        A function to use the mean square displacement of the dipole moment of a system to extract the
        ionic conductivity

        Parameters
        ----------
        data_range : int
                Time range over which the measurement should be performed
        plot : bool
                If True, will plot the MSD over time
        """

        # Instantiate the calculator
        calculation_ehic = einstein_helfand_ionic_conductivity.EinsteinHelfandIonicConductivity(self,
                                                                                                data_range=data_range,
                                                                                                plot=plot)
        calculation_ehic.run_analysis()  # Perform analysis.
        self.save_class()               # Update the class state.

    def green_kubo_ionic_conductivity(self, data_range, plot=True):
        """
        Calculate Green-Kubo Ionic Conductivity

        A function to use the current autocorrelation function to calculate the Green-Kubo ionic conductivity of the
        system being studied.

        Parameters
        ----------
        data_range : int
                Number of data points with which to calculate the conductivity

        plot : bool
                If True, a plot of the current autocorrelation function will be generated
        """

        # Instantiate the calculator
        calculation_gkic = green_kubo_ionic_conductivity.GreenKuboIonicConductivity(self, plot=plot,
                                                                                    data_range=data_range)
        calculation_gkic.run_analysis()  # run the analysis
        self.save_class()               # Update class state

    def radial_distribution_function(self, plot=True, bins=500, cutoff=None, start=0, stop=None, n_confs=1000,
                                     n_batches=1):
        """
        Calculate the radial distribution function

        Parameters
        ----------
        plot : bool
                If true, the analysis plots will be saved.
        bins : int
                Number of bins to use in the histogram.
        cutoff : float
                Cutoff to apply to the calculation. Should be <= half the box size as we have not written a generalized
                density calculator.
        start: int
                Starting configuration.
        stop : int
                Final confguration to include.
        n_confs : int
                Number of configurations to use.
        n_batches : int
                Number of batches to use. TODO: Make this a calculated property during analysis.
        """

        # Instantiate the calculator
        calculation_rdf = radial_distribution_function.RadialDistributionFunction(self, plot=plot,
                                                                                  bins=bins,
                                                                                  cutoff=cutoff,
                                                                                  start=start,
                                                                                  stop=stop,
                                                                                  n_confs=n_confs,
                                                                                  n_batches=n_batches)
        calculation_rdf.run_analysis()                  # run the analysis
        self.radial_distribution_function_state = True  # update the analysis state
        self.save_class()                              # save the class state

    def calculate_coordination_numbers(self, plot=True):
        """
        Calculate the coordination numbers

        Parameters
        ----------
        plot : bool
                If true, images of the calculation will be saved.
        """

        calculation_cn = coordination_number_calculation.CoordinationNumbers(self, plot)  # Instantiate the calculator
        calculation_cn.run_analysis()                                                     # Run the analysis
        self.save_class()                                                                # Update the class state

    def potential_of_mean_force(self, plot=True, save=True):
        """
        Calculate the potential of mean-force

        Parameters
        ----------
        plot : bool
                If true, plots of the analysis will be saved
        save : bool
                If true, data calculated during the analysis will be saved.
        """

        calculation_pomf = potential_of_mean_force.PotentialOfMeanForce(self, plot=plot, save=save)  # Load calculator
        calculation_pomf.run_analysis()                                                              # Perform analysis
        self.save_class()                                                                           # Update class

    def kirkwood_buff_integrals(self, plot=True, save=True):
        """
        Calculate the kirkwood buff integrals

        Parameters
        ----------
        plot : bool
                If true, plots of the analysis will be saved
        save : bool
                If true, data associated with the calculation will be saved.
        """

        calculation_kbi = kirkwood_buff_integrals.KirkwoodBuffIntegral(self, plot=plot, save=save)  # Load calculator
        calculation_kbi.run_analysis()                                                              # Perform analysis
        self.kirkwood_buff_integral_state = True                                                    # update class
        self.save_class()                                                                          # Save class state

    def structure_factor(self):
        """
        Calculate the structure factor
        """

        calculation_strfac = structure_factor.StructureFactor(self)  # Instantiate calculator class
        calculation_strfac.run_analysis()                            # Run the analysis
        self.save_class()                                           # Update the class state.

    # TODO def green_kubo_viscosity(self):
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
        self.save_class()

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
        """
        Dump a json file.

        Returns
        -------

        """
        filename = Path(f"{self.storage_path}/{self.analysis_name}.json")
        with open(filename, 'w') as fp:
            json.dump(self.results, fp, indent=4, sort_keys=True)
