"""
The central experiment class fundamental to all analysis.

Summary
-------
The experiment class is the main class involved in characterizing and analyzing a simulation.
"""

import json
import os
import pickle
import sys
#from importlib.resources import open_text
from pathlib import Path

import h5py as hf
import numpy as np
import pubchempy as pcp
import tensorflow as tf
import yaml
from tqdm import tqdm

from mdsuite import data as static_data
from mdsuite.calculators.computations_dict import dict_classes_computations
from mdsuite.transformations.transformation_dict import transformations_dict
from mdsuite.file_io.file_io_dict import dict_file_io
from mdsuite.utils.units import units_dict
from mdsuite.utils.exceptions import *
from mdsuite.database.database import Database
from mdsuite.file_io.file_read import FileProcessor


class Experiment:
    """
    The central experiment class fundamental to all analysis.

    Attributes
    ----------
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
   """

    def __init__(self, analysis_name, storage_path='./', time_step=1.0, temperature=0, units='real'):
        """
        Initialise the experiment class.

        Attributes
        ----------
        analysis_name : str
                The name of the analysis being performed e.g. NaCl_1400K
        storage_path : str
                Path to where the data should be stored (best to have  drive capable of storing large files)
        temperature : float
                The temperature of the simulation that should be used in some analysis.
        time_step : float
                Time step of the simulation e.g 0.002. Necessary as it cannot be easily read in from the trajectory.
        """

        # Taken upon instantiation
        self.analysis_name = analysis_name  # Name of the experiment.
        self.storage_path = os.path.abspath(storage_path)  # Where to store the data
        self.temperature = temperature  # Temperature of the system.
        self.time_step = time_step  # Timestep chosen for the simulation.

        # Added from trajectory file
        self.units = self.units_to_si(units)  # Units used during the simulation.
        self.number_of_configurations = 0  # Number of configurations in the trajectory.
        self.number_of_atoms = None  # Number of atoms in the simulation.
        self.species = None  # Species dictionary.
        self.box_array = None  # Box vectors.
        self.dimensions = None  # Dimensionality of the system.

        self.sample_rate = None  # Rate at which configurations are dumped in the trajectory.
        self.batch_size = None  # Number of configurations in each batch.
        self.volume = None  # Volume of the system.
        self.properties = None  # Properties measured in the simulation.
        self.property_groups = None  # Names of the properties measured in the simulation

        # Internal File paths
        self.experiment_path = os.path.join(self.storage_path, self.analysis_name)  # path to the experiment files
        self.database_path = os.path.join(self.experiment_path, 'databases')  # path to the databases
        self.figures_path = os.path.join(self.experiment_path, 'figures')  # path to the figures directory

        self.radial_distribution_function_state = False  # Set true if this has been calculated
        self.kirkwood_buff_integral_state = False  # Set true if it has been calculated
        self.structure_factor_state = False

        self.results = [
            'diffusion_coefficients',
            'ionic_conductivity',
            'thermal_conductivity',
            'coordination_numbers',
            'potential_of_mean_force_values',
            'radial_distribution_function',
            'kirkwood_buff_integral'
        ]

        # Memory properties
        self.memory_requirements = {}

        # Check if the experiment exists and load if it does.
        self._load_or_build()

        # Run Computations
        self.run_computation = self.RunComputation(self)

    def _load_or_build(self):
        """
        Check if the experiment already exists and decide whether to load it or build a new one.
        """

        # Check if the experiment exists and load if it does.
        if Path(self.experiment_path).exists():
            print("This experiment already exists! I'll load it up now.")
            self.load_class()
        else:
            print("Creating a new experiment! How exciting!")
            self._build_model()

    def load_class(self):
        """
        Load class instance

        A function to load a class instance given the project name.
        """

        class_file = open(f'{self.storage_path}/{self.analysis_name}/{self.analysis_name}.bin', 'rb')  # open file
        pickle_data = class_file.read()  # read file
        class_file.close()  # close file

        self.__dict__ = pickle.loads(pickle_data)  # update the class object

    def save_class(self):
        """
        Saves class instance

        In order to keep properties of a class the state must be stored. This method will store the instance of the
        class for later re-loading
        """

        save_file = open(os.path.join(self.experiment_path, f"{self.analysis_name}.bin"), 'wb')  # construct file
        save_file.write(pickle.dumps(self.__dict__))  # write to file
        save_file.close()  # close the file

    def units_to_si(self, units_system):
        """
        Returns a dictionary with equivalences from the unit system given by a string to SI.
        Along with some constants in the unit system provided (boltzman, or other conversions).
        Instead, the user may provide a dictionary. In that case, the dictionary will be used as the unit system.


        Parameters
        ----------
        units_system (str) -- current unit system
        dimension (str) -- dimension you would like to change

        Returns
        -------
        conv_factor (float) -- conversion factor to pass to SI

        Examples
        --------
        Pass from metal units of time (ps) to SI

        >>> self.units_to_si('metal')
        {'time': 1e-12, 'length': 1e-10, 'energy': 1.6022e-19}
        >>> self.units_to_si({'time': 1e-12, 'length': 1e-10, 'energy': 1.6022e-19,
        'NkTV2p':1.6021765e6, 'boltzman':8.617343e-5})
        """

        if isinstance(units_system, dict):
            return units_system
        else:
            try:
                units = units_dict[units_system]()  # executes the function to return the appropriate dictionary.
            except KeyError:
                print(f'The unit system provided is not implemented...')
                print(f'The available systems are: ')
                [print(key) for key, _ in units_dict.items()]
                sys.exit(-1)
        return units

    class RunComputation:
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

        def __init__(self, parent):
            self.parent = parent
            for key in dict_classes_computations:
                self.__setattr__(key, dict_classes_computations[key])

        def __getattribute__(self, item):
            """Call via function
            You cann call the computation via a function and autocompletion
            >>> Experiment.run_computation.EinsteinDiffusionCoefficients(plot=True)
            """
            try:
                class_compute = dict_classes_computations[item]
            except KeyError:
                return super().__getattribute__(item)

            def func(**kwargs):
                self.compute(class_compute, **kwargs)

            return func

        def __call__(self, computation_name, **kwargs):
            """Call directly
            You can call the computation directly via
            >>> Experiment.run_computation("EinsteinDiffusionCoefficients", plot=True)
            """
            try:
                class_compute = dict_classes_computations[computation_name]
            except KeyError:
                print(f'{computation_name} not found')
                print(f'Available computations are:')
                [print(key) for key in dict_classes_computations.keys()]
                return

            self.compute(class_compute, **kwargs)

        def compute(self, class_compute, **kwargs):
            object_compute = class_compute(self.parent, **kwargs)
            object_compute.run_analysis()
            self.parent.save_class()

    def run_computation(self, computation_name, **kwargs):

        try:
            class_compute = dict_classes_computations[computation_name]
        except KeyError:
            print(f'{computation_name} not found')
            print(f'Available computations are:')
            [print(key) for key in dict_classes_computations.keys()]
            return

        object_compute = class_compute(self, **kwargs)
        object_compute.run_analysis()
        self.save_class()

    def perform_transformation(self, transformation_name, **kwargs):
        """
        Perform a transformation on the system.

        Parameters
        ----------
        transformation_name : str
                Name of the transformation to perform.
        **kwargs
                Other arguments associated with the transformation.

        Returns
        -------
        Update of the database.
        """

        try:
            transformation = transformations_dict[transformation_name]
        except KeyError:
            print(f'{transformation_name} not found')
            print(f'Available transformations are:')
            [print(key) for key in transformations_dict.keys()]
            return

        transformation_run = transformation(self, **kwargs)
        transformation_run.run_transformation()  # perform the transformation

    def _build_model(self):
        """
        Build the 'experiment' for the analysis

        A method to build the database in the hdf5 format. Within this method, several other are called to develop the
        database skeleton, get configurations, and process and store the configurations. The method is accompanied
        by a loading bar which should be customized to make it more interesting.
        """

        # Create new analysis directory and change into it
        try:
            os.mkdir(self.experiment_path)  # Make the experiment directory
            os.mkdir(self.figures_path)  # Create a directory to save images
            os.mkdir(self.database_path)  # Create a directory for data

        except FileExistsError:  # throw exception if the file exits
            return

        self.save_class()  # save the class state.
        print(f"** An experiment has been added titled {self.analysis_name} **")

    def print_class_attributes(self):
        """
        Print all attributes of the class

        Returns
        -------
        attributes : list
                List of class attribute tuples of (key, value)
        """

        attributes = []  # define empty array
        for item in vars(self).items():  # loop over class attributes
            attributes.append(item)  # append to the attributes array
        for tuple_attributes in attributes:  # Split the key and value terms
            print(f"{tuple_attributes[0]}: {tuple_attributes[1]}")  # Format the print statement

        return attributes

    def print_data_structure(self):
        """
        Print the data structure of the hdf5 dataset
        """

        database = hf.File(os.path.join(self.database_path, 'database.hdf5'), "r")
        with Diagram("Web Service", show=True, direction='TB'):
            head = RDS("Database")  # set the head database object
            for item in database:
                with Cluster(f"{item}"):
                    group_list = []
                    for property_group in database[item]:
                        group_list.append(ECS(property_group))  # construct a list of equal level groups
                head >> group_list  # append these groups to the head object

    def add_data(self, trajectory_file=None, file_format='lammps_traj', rename_cols=None):
        """
        Add data to the database

        Parameters
        ----------
        file_format :
                Format of the file being read in. Default is file_path
        trajectory_file : str
                Trajectory file to be process and added to the database.
        """

        # Check if there is a trajectory file.
        if trajectory_file is None:
            print("No data has been given")
            return  # exit method as nothing more can be done

        # Load the file reader and the database object
        trajectory_reader, file_type = self._load_trajectory_reader(file_format, trajectory_file)
        database = Database(name=os.path.join(self.database_path, "database.hdf5"), architecture='simulation')

        # Check to see if a database exists
        database_path = Path(os.path.join(self.database_path, 'database.hdf5'))  # get theoretical path.

        if database_path.exists():
            pass  # self._update_database(trajectory_reader)
        else:
            self._build_new_database(trajectory_reader, trajectory_file, database)

        self.memory_requirements = database.get_memory_information()
        self.save_class()  # Update the class state.

    def _build_new_database(self, trajectory_reader: FileProcessor, trajectory_file, database):
        """
        Build a new database
        """

        architecture, line_length = trajectory_reader.process_trajectory_file()  # get properties of the trajectory file
        database.initialize_database(architecture)  # initialize the database
        db_object = database.open()  # Open a database object
        batch_range = int(self.number_of_configurations / self.batch_size)  # calculate the batch range
        counter = 0  # instantiate counter
        structure = trajectory_reader.build_file_structure()  # build the file structure

        f_object = open(trajectory_file, 'r')  # open the trajectory file
        for _ in tqdm(range(batch_range)):
            database.add_data(data=trajectory_reader.read_configurations(self.batch_size, f_object, line_length),
                              structure=structure,
                              database=db_object,
                              start_index=counter,
                              batch_size=self.batch_size)
            counter += self.batch_size

        database.close(db_object)  # Close the object
        f_object.close()

        # Build database for analysis output
        with hf.File(os.path.join(self.database_path, "analysis_data.hdf5"), "w") as db:
            for key in self.results:
                db.create_group(key)

        # Instantiate YAML file for system properties
        with open(os.path.join(self.database_path, 'system_properties.yaml'), 'w') as f:
            data = {'diffusion_coefficients': {'einstein_diffusion_coefficients': {'Singular': {}, 'Distinct': {}},
                                               'Green_Kubo_Diffusion': {'Singular': {}, 'Distinct': {}}},
                    'ionic_conductivity': {},
                    'thermal_conductivity': {},
                    'coordination_numbers': {'Coordination_Numbers': {}},
                    'potential_of_mean_force_values': {'Potential_of_Mean_Force': {}},
                    'radial_distribution_function': {},
                    'kirkwood_buff_integral': {},
                    'structure_factor': {},
                    'viscosity':{}}

            yaml.dump(data, f)

        self.save_class()  # Update the class state

    def _load_trajectory_reader(self, file_format, trajectory_file):
        try:
            class_file_io, file_type = dict_file_io[file_format]  # file type is per atoms or flux.
        except KeyError:
            print(f'{file_format} not found')
            print(f'Available io formats are are:')
            [print(key) for key in dict_file_io.keys()]
            sys.exit(1)
        return class_file_io(self, file_path=trajectory_file), file_type

    def build_species_dictionary(self):
        """
        Add information to the species dictionary

        A fundamental part of this package is species specific analysis. Therefore, the Pubchempy package is
        used to add important species specific information to the class. This will include the charge of the ions which
        will be used in conductivity calculations.

        """
        with importlib.resources.open_text(static_data, 'PubChemElements_all.json') as json_file:
            pse = json.loads(json_file.read())

        # Try to get the species data from the Periodic System of Elements file
        for element in self.species:
            self.species[element]['charge'] = [0.0]
            for entry in pse:
                if pse[entry][1] == element:
                    self.species[element]['mass'] = [float(pse[entry][3])]

        # If gathering the data from the PSE file was not successful try to get it from Pubchem via pubchempy
        for element in self.species:
            if 'mass' not in self.species[element]:
                try:
                    temp = pcp.get_compounds(element, 'name')
                    temp[0].to_dict(properties=['atoms', 'bonds', 'exact_mass', 'molecular_weight', 'elements'])
                    self.species[element]['mass'] = temp[0].molecular_weight
                    print(temp[0].exact_mass)
                except (ElementMassAssignedZero, IndexError):
                    self.species[element]['mass'] = [0.0]
                    print(f'WARNING element {element} has been assigned mass=0.0')

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
        if not species:
            species = list(self.species.keys())  # get list of all species available.

        # If no slice is given, load all configurations.
        if select_slice is None:
            select_slice = np.s_[:]  # set the numpy slice object.

        property_matrix = []  # Define an empty list for the properties to fill

        with hf.File(os.path.join(self.database_path, 'database.hdf5'), "r+") as database:
            for item in list(species):
                path = os.path.join(item, identifier)

                # Unwrap the positions if they need to be unwrapped
                if identifier == "Unwrapped_Positions" and "Unwrapped_Positions" not in database[item]:
                    print("We first have to unwrap the coordinates... Doing this now")
                    self.perform_transformation('UnwrapCoordinates', species=[item])  # Unwrap the coordinates

                # Check if the desired property is in the database.
                if identifier not in database[item]:
                    print("This data was not found in the database. Was it included in your simulation input?")
                    return

                # If the tensor kwarg is True, return a tensor.
                if tensor:
                    property_matrix.append(
                        tf.convert_to_tensor(database[path][select_slice], dtype=tf.float64))

                elif sym_matrix:  # return a stress tensor
                    property_matrix.append(database[path][select_slice])
                elif scalar:  # return a scalar
                    property_matrix.append(database[path][select_slice])

                else:  # return a numpy array
                    property_matrix.append(database[path][select_slice])

        # Check if the property loaded was a scalar.
        if len(property_matrix) == 1:
            return property_matrix[0]  # return the scalar dataset
        else:
            return property_matrix  # return the full tensor object.
