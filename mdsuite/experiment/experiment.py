"""
The central experiment class fundamental to all analysis.

Summary
-------
The experiment class is the main class involved in characterizing and analyzing a simulation.
"""
import logging

import json
import os
import pickle
import sys
from pathlib import Path
import inspect

import h5py as hf
import numpy as np
import pubchempy as pcp
import yaml
from tqdm import tqdm
import importlib.resources

from datetime import datetime

from mdsuite.calculators.computations_dict import dict_classes_computations, dict_classes_db
from mdsuite.transformations.transformation_dict import transformations_dict
from mdsuite.file_io.file_io_dict import dict_file_io
from mdsuite.utils.units import units_dict
from mdsuite.utils.meta_functions import join_path
from mdsuite.utils.exceptions import *
from mdsuite.database.database import Database
from mdsuite.file_io.file_read import FileProcessor
from mdsuite.visualizer.visualizer import TrajectoryVisualizer
from mdsuite.database.properties_database import PropertiesDatabase


class Experiment:
    """
    The central experiment class fundamental to all analysis.

    Attributes
    ----------
    analysis_name : str
            The name of the analysis being performed e.g. NaCl_1400K
    storage_path : str
            Path to where the tensor_values should be stored (best to have  drive capable of storing large files)
    temperature : float
            The temperature of the simulation that should be used in some analysis. Necessary as it cannot be easily
            read in from the simulation tensor_values.
    time_step : float
            Time step of the simulation e.g 0.002. Necessary as it cannot be easily read in from the trajectory.
    volume : float
            Volume of the simulation box
    species : dict
            A dictionary of the species in the experiment and their properties. Their properties includes
            index location in the trajectory file, mass of the species as taken from the PubChem
            database_path, and the charge taken from the same database_path. When using these properties, it is
            best that users confirm this information, with exception to the indices as they are read
            from the file and will be correct.
    number_of_atoms : int
            The total number of atoms in the simulation
   """

    def __init__(self, analysis_name, storage_path='./', time_step=1.0, temperature=0, units='real',
                 cluster_mode=False):
        """
        Initialise the experiment class.

        Attributes
        ----------
        analysis_name : str
                The name of the analysis being performed e.g. NaCl_1400K
        storage_path : str
                Path to where the tensor_values should be stored (best to have  drive capable of storing large files)
        temperature : float
                The temperature of the simulation that should be used in some analysis.
        time_step : float
                Time step of the simulation e.g 0.002. Necessary as it cannot be easily read in from the trajectory.
        cluster_mode : bool
                If true, several parameters involved in plotting and parallelization will be adjusted so as to allow
                for optimal performance on a large computing cluster.
        """

        # Taken upon instantiation
        self.analysis_name = analysis_name  # Name of the experiment.
        self.storage_path = os.path.abspath(storage_path)  # Where to store the tensor_values
        self.temperature = temperature  # Temperature of the experiment.
        self.time_step = time_step  # Timestep chosen for the simulation.
        self.cluster_mode = cluster_mode  # whether or not the script will run on a cluster

        # Added from trajectory file
        self.units = self.units_to_si(units)  # Units used during the simulation.
        self.number_of_configurations = 0  # Number of configurations in the trajectory.
        self.number_of_atoms = None  # Number of atoms in the simulation.
        self.species = None  # Species dictionary.
        self.molecules = {}  # molecules
        self.box_array = None  # Box vectors.
        self.dimensions = None  # Dimensionality of the experiment.

        self.sample_rate = None  # Rate at which configurations are dumped in the trajectory.
        self.batch_size = None  # Number of configurations in each batch.
        self.volume = None  # Volume of the experiment.
        self.properties = None  # Properties measured in the simulation.
        self.property_groups = None  # Names of the properties measured in the simulation

        # Internal File paths
        self.experiment_path: str
        self.database_path: str
        self.figures_path: str
        self.logfile_path: str
        self._create_internal_file_paths()  # fill the path attributes

        self.radial_distribution_function_state = False  # Set true if this has been calculated
        self.kirkwood_buff_integral_state = False  # Set true if it has been calculated
        self.structure_factor_state = False

        self._results = list(dict_classes_db.keys())

        # Memory properties
        self.memory_requirements = {}  # TODO I think this can be removed. - Not until all calcs are under tf Dataset

        # Check if the experiment exists and load if it does.
        self._load_or_build()

        # Run Computations
        self.run_computation = self.RunComputation(self)

        self.log = None
        self._start_logging()

    def _start_logging(self):
        logfile_name = datetime.now().replace(microsecond=0).isoformat().replace(':', '-') + ".log"
        logfile = os.path.join(self.logfile_path, logfile_name)

        # there is a good chance, that the root logger is already defined so we have to make some changes to it!
        root = logging.getLogger()

        try:
            if not hasattr(root.handlers[0], 'baseFilename'):
                # we remove all existing handlers, to avoid messages being written multiple times,
                # but we only do this when it is not a file handler, because we define the file handler first, so we
                # know that there already exist one and this might be a second experiment that has been loaded!
                while root.hasHandlers():
                    root.removeHandler(root.handlers[0])
        except IndexError:
            # No handlers available, so we have a new root handler
            pass

        root.setLevel(logging.DEBUG)  # <- seems to be the lowest possible loglevel so we set it to debug here!
        # if we set it to info, we can not set the file oder stream to debug

        # Logging to the logfile
        file_handler = logging.FileHandler(filename=logfile)
        file_handler.setLevel(logging.INFO)  # <- file log loglevel

        formatter = logging.Formatter('%(asctime)s - %(name)s (%(levelname)s) - %(message)s')
        file_handler.setFormatter(formatter)
        # attaching the stdout handler to the configured logging
        root.addHandler(file_handler)

        sys_handler = False
        for handler in root.handlers:
            if not hasattr(handler, 'baseFilename'):  # assume that  every handler is filehandler except for sys handler
                sys_handler = True

        if not sys_handler:
            # Logging to the stdout (Terminal, Jupyter, ...)
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(logging.INFO)  # <- stdout loglevel

            formatter = logging.Formatter('%(asctime)s - %(name)s (%(levelname)s) - %(message)s')
            stream_handler.setFormatter(formatter)
            # attaching the stdout handler to the configured logging
            root.addHandler(stream_handler)

        # get the file specific logger to get information where the log was written!
        self.log = logging.getLogger(__name__)
        self.log.info(f"Created logfile {logfile_name} in experiment path {self.logfile_path}")

    def _create_internal_file_paths(self):
        """
        Create or update internal file paths
        """
        self.experiment_path = os.path.join(self.storage_path, self.analysis_name)  # path to the experiment files
        self.database_path = os.path.join(self.experiment_path, 'databases')  # path to the databases
        self.figures_path = os.path.join(self.experiment_path, 'figures')  # path to the figures directory
        self.logfile_path = os.path.join(self.experiment_path, 'logfiles')

    def _load_or_build(self) -> bool:
        """
        Check if the experiment already exists and decide whether to load it or build a new one.
        """

        # Check if the experiment exists and load if it does.
        if Path(self.experiment_path).exists():
            print("This experiment already exists! I'll load it up now.")
            # Can not log this to a file, because we don't know the file path yet!
            self.load_class()
            return True
        else:
            print("Creating a new experiment! How exciting!")
            # Can not log this to a file, because we don't know the file path yet!
            self._build_model()
            return False

    def load_class(self):
        """
        Load class instance

        A function to load a class instance given the project name.
        """

        storage_path = self.storage_path  # store the new storage path

        with open(f'{self.storage_path}/{self.analysis_name}/{self.analysis_name}.bin', 'rb') as f:
            self.__dict__ = pickle.loads(f.read())

        self.storage_path = storage_path  # set the one form the database to the new one
        self._create_internal_file_paths()  # force rebuild every time

        self.run_computation = self.RunComputation(self)
        self._start_logging()
        # TODO Why is this necessary? What does it exactly?

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
        Returns a dictionary with equivalences from the unit experiment given by a string to SI.
        Along with some constants in the unit experiment provided (boltzman, or other conversions).
        Instead, the user may provide a dictionary. In that case, the dictionary will be used as the unit experiment.


        Parameters
        ----------
        units_system (str) -- current unit experiment
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
                print(f'The unit experiment provided is not implemented...')
                print(f'The available systems are: ')
                [print(key) for key, _ in units_dict.items()]
                sys.exit(-1)
        return units

    def map_elements(self, mapping: dict = None):
        """
        Map numerical keys to element names in the Experiment class and database_path.

        Returns
        -------
        Updates the class
        """

        if mapping is None:
            # self.log("Must provide a mapping")
            self.log.info("Must provide a mapping")
            return

        # rename keys in species dictionary
        for item in mapping:
            self.species[mapping[item]] = self.species.pop(item)

        # rename database_path groups
        db_object = Database(name=os.path.join(self.database_path, "database_path.hdf5"))
        db_object.change_key_names(mapping)

        self.save_class()  # update the class state

    class RunComputation:
        """ Run a computation

        The type of computation will be stored in a dictionary.

        This class represents the "run_computation" in "run_computation.calculator"

        Parameters
        ----------
        parent : object
                Experiment class in which this class is contained.

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
            You can call the computation via a function and autocompletion
            >>> self.run_computation.EinsteinDiffusionCoefficients(plot=True)
            """
            try:
                class_compute = dict_classes_computations[item]
            except KeyError:
                return super().__getattribute__(item)

            class Func:
                """Return the documentation if the function is not called.

                This class represents the "calculator" in "run_computation.calculator"

                """

                def __init__(self, parent, class_func_compute):
                    self.parent = parent
                    self.class_compute = class_func_compute

                def get_documentation(self):
                    """
                    Get the documentation for the calculator
                    You can print the documentation via
                    self.run_computation.EinsteinDiffusionCoefficients.get_documentation()
                    """
                    print(inspect.getdoc(self.class_compute))

                def __repr__(self):
                    """
                    Get the documentation for the calculator
                    You can print the documentation if you don't call the class
                    self.run_computation.EinsteinDiffusionCoefficients
                    """
                    self.get_documentation()
                    return f"Please use Experiment.run_computation.calculator(*args, **kwargs) to run the calculation"

                def __call__(self, **kwargs):
                    """
                    Introduce call method.
                    """
                    output = self.parent.compute(self.class_compute, **kwargs)
                    return output

            return Func(self, class_compute)

        def __call__(self, computation_name, **kwargs):
            """Call directly
            You can call the computation directly via
            >>> self.run_computation("EinsteinDiffusionCoefficients", plot=True)
            """
            try:
                class_compute = dict_classes_computations[computation_name]
            except KeyError:
                print(f'{computation_name} not found')
                print(f'Available computations are:')
                [print(key) for key in dict_classes_computations.keys()]
                return

            return self.compute(class_compute, **kwargs)

        def __repr__(self):
            """Print available computations if no computation method is called
            """
            return_string = 'Available computations are: \n \n'
            for key in dict_classes_computations:
                return_string += key
                return_string += "\n"
            return return_string

        def compute(self, class_compute, **kwargs):
            """
            A doc string that should have been added by the person who wrote the method.
            Parameters
            ----------
            class_compute
            kwargs

            Returns
            -------

            """
            object_compute = class_compute(self.parent, **kwargs)
            dat = object_compute.run_analysis()
            self.parent.save_class()
            return dat

    def perform_transformation(self, transformation_name, **kwargs):
        """
        Perform a transformation on the experiment.

        Parameters
        ----------
        transformation_name : str
                Name of the transformation to perform.
        **kwargs
                Other arguments associated with the transformation.

        Returns
        -------
        Update of the database_path.
        """

        try:
            transformation = transformations_dict[transformation_name]
        except KeyError:
            print(f'{transformation_name} not found')
            print(f'Available transformations are:')
            [print(key) for key in transformations_dict.keys()]
            sys.exit(1)

        transformation_run = transformation(self, **kwargs)
        transformation_run.run_transformation()  # perform the transformation

    def _build_model(self):
        """
        Build the 'experiment' for the analysis

        A method to build the database_path in the hdf5 format. Within this method, several other are called to develop the
        database_path skeleton, get configurations, and process and store the configurations. The method is accompanied
        by a loading bar which should be customized to make it more interesting.
        """

        # Create new analysis directory and change into it
        try:
            os.mkdir(self.experiment_path)  # Make the experiment directory
            os.mkdir(self.figures_path)  # Create a directory to save images
            os.mkdir(self.database_path)  # Create a directory for tensor_values
            os.mkdir(self.logfile_path)

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
            self.log.info(f"{tuple_attributes[0]}: {tuple_attributes[1]}")

        return attributes

    def add_data(self, trajectory_file: str = None, file_format: str = 'lammps_traj', rename_cols: dict = None,
                 sort: bool = False):
        """
        Add tensor_values to the database_path

        Parameters
        ----------
        file_format :
                Format of the file being read in. Default is file_path
        trajectory_file : str
                Trajectory file to be process and added to the database_path.
        rename_cols : dict
                If this argument is given, the columns with names in the keys of the dictionary will be replaced with
                the values.
        sort : bool
                If true, the tensor_values will be sorted when being entered into the database_path.
        """

        # Check if there is a trajectory file.
        if trajectory_file is None:
            print("No tensor_values has been given")
            sys.exit(1)

        # Load the file reader and the database_path object
        trajectory_reader, file_type = self._load_trajectory_reader(file_format, trajectory_file, sort=sort)
        database = Database(name=os.path.join(self.database_path, "database.hdf5"), architecture='simulation')

        # Check to see if a database_path exists
        database_path = Path(os.path.join(self.database_path, 'database.hdf5'))  # get theoretical path.

        if file_type == 'flux':
            flux = True
        else:
            flux = False

        if database_path.exists():
            self._update_database(trajectory_reader, trajectory_file, database, rename_cols)

        else:
            self._build_new_database(trajectory_reader, trajectory_file, database, rename_cols=rename_cols, flux=flux, sort=sort)

        self.build_species_dictionary()
        self.memory_requirements = database.get_memory_information()
        self.save_class()  # Update the class state.

    def _build_new_database(self, trajectory_reader: FileProcessor, trajectory_file: str, database: Database,
                            rename_cols: dict, flux: bool = False, sort: bool =False):
        """
        Build a new database_path
        """
        # get properties of the trajectory file
        architecture, line_length = trajectory_reader.process_trajectory_file(rename_cols=rename_cols)
        database.initialize_database(architecture)  # initialize the database_path

        batch_range = int(self.number_of_configurations / self.batch_size)  # calculate the batch range
        remainder = self.number_of_configurations - batch_range * self.batch_size
        counter = 0  # instantiate counter
        structure = trajectory_reader.build_file_structure()  # build the file structure

        f_object = open(trajectory_file, 'r')  # open the trajectory file
        for _ in tqdm(range(batch_range), ncols=70):
            database.add_data(data=trajectory_reader.read_configurations(self.batch_size, f_object, line_length),
                              structure=structure,
                              start_index=counter,
                              batch_size=self.batch_size,
                              flux=flux,
                              n_atoms=self.number_of_atoms, sort=sort)
            counter += self.batch_size

        if remainder > 0:
            structure = trajectory_reader.build_file_structure(batch_size=remainder)  # build the file structure
            database.add_data(data=trajectory_reader.read_configurations(remainder, f_object, line_length),
                              structure=structure,
                              start_index=counter,
                              batch_size=remainder,
                              flux=flux, sort=sort)

        f_object.close()

        # Build database_path for analysis output
        with hf.File(os.path.join(self.database_path, "analysis_data.hdf5"), "w") as db:
            for key in self._results:
                db.create_group(key)

        # Instantiate YAML file for experiment properties
        with open(os.path.join(self.database_path, 'system_properties.yaml'), 'w') as f:
            data = dict_classes_db

            yaml.dump(data, f)

        property_database = PropertiesDatabase(name=os.path.join(self.database_path, "property_database"))
        property_database.build_database()

        self.save_class()  # Update the class state

    def _update_database(self, trajectory_reader: FileProcessor, trajectory_file: str, database: Database,
                         rename_cols: dict, flux: bool = False):
        """
        Update the database rather than build a new database.

        Returns
        -------
        Updates the current database.
        """
        counter = self.number_of_configurations
        architecture, line_length = trajectory_reader.process_trajectory_file(rename_cols=rename_cols)
        number_of_new_configurations = self.number_of_configurations - counter
        database.resize_dataset(architecture)  # initialize the database_path
        batch_range = int(number_of_new_configurations / self.batch_size)  # calculate the batch range
        remainder = number_of_new_configurations - (batch_range * self.batch_size)
        structure = trajectory_reader.build_file_structure()  # build the file structure
        f_object = open(trajectory_file, 'r')  # open the trajectory file
        for _ in tqdm(range(batch_range), ncols=70):
            database.add_data(data=trajectory_reader.read_configurations(self.batch_size, f_object, line_length),
                              structure=structure,
                              start_index=counter,
                              batch_size=self.batch_size,
                              flux=flux,
                              n_atoms=self.number_of_atoms)
            counter += self.batch_size

        if remainder > 0:
            structure = trajectory_reader.build_file_structure(batch_size=remainder)  # build the file structure
            database.add_data(data=trajectory_reader.read_configurations(remainder, f_object, line_length),
                              structure=structure,
                              start_index=counter,
                              batch_size=remainder,
                              flux=flux)

        f_object.close()

    def _load_trajectory_reader(self, file_format, trajectory_file, sort: bool = False):
        try:
            class_file_io, file_type = dict_file_io[file_format]  # file type is per atoms or flux.
        except KeyError:
            print(f'{file_format} not found')
            print(f'Available io formats are are:')
            [print(key) for key in dict_file_io.keys()]
            sys.exit(1)
        return class_file_io(self, file_path=trajectory_file, sort=sort), file_type

    def build_species_dictionary(self):
        """
        Add information to the species dictionary

        A fundamental part of this package is species specific analysis. Therefore, the Pubchempy package is
        used to add important species specific information to the class. This will include the charge of the ions which
        will be used in conductivity calculations.

        """
        with importlib.resources.open_text("mdsuite.data", 'PubChemElements_all.json') as json_file:
            pse = json.loads(json_file.read())

        # Try to get the species tensor_values from the Periodic System of Elements file
        for element in self.species:
            self.species[element]['charge'] = [0.0]
            for entry in pse:
                if pse[entry][1] == element:
                    self.species[element]['mass'] = [float(pse[entry][3])]

        # If gathering the tensor_values from the PSE file was not successful try to get it from Pubchem via pubchempy
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
        self.save_class()

    def set_element(self, old_name, new_name):
        """
        Change the name of the element in the self.species dictionary

        Parameters
        ----------
        old_name : str
                Name of the element you want to change
        new_name : str
                New name of the element
        """
        # Check if the new name is new
        if new_name != old_name:
            self.species[new_name] = self.species[old_name]  # update dict
            del self.species[old_name]  # remove old entry

    def set_charge(self, element, charge):
        """
        Set the charge/s of an element

        Parameters
        ----------
        element : str
                Name of the element whose charge you want to change
        charge : list
                New charge/s of the element
        """
        self.species[element]['charge'] = charge  # update entry

    def set_mass(self, element, mass):
        """
        Set the mass/es of an element

        Parameters
        ----------
        element : str
                Name of the element whose mass you want to change
        mass : list
                New mass/es of the element
        """
        self.species[element]['mass'] = mass  # update the mass

    def run_visualization(self, species: list = None, unwrapped: bool = False):
        """
        Run a visualization on the database.

        Parameters
        ----------
        species : list
                List of species you wish to visualize.
        unwrapped : bool
                If true, unwrapped coordinates will be used.
        Returns
        -------
        """
        if species is None:
            species = list(self.species)
        visualizer = TrajectoryVisualizer(experiment=self, species=species, unwrapped=unwrapped)
        visualizer.run_visualization()

    def load_matrix(self, identifier: str = None, species: dict = None, select_slice: np.s_ = None, path: list = None):
        """
        Load a desired property matrix.

        Parameters
        ----------
        identifier : str
                Name of the matrix to be loaded, e.g. Unwrapped_Positions, Velocities
        species : list
                List of species to be loaded
        select_slice : np.slice
                A slice to select from the database_path.
        path : str
                optional path to the database_path.

        Returns
        -------
        property_matrix : np.array, tf.tensor
                Tensor of the property to be studied. Format depends on kwargs.
        """
        database = Database(name=os.path.join(self.database_path, 'database.hdf5'))

        if path is not None:
            return database.load_data(path_list=path, select_slice=select_slice)

        else:
            # If no species list is given, use all species in the Experiment class instance.
            if species is None:
                species = list(self.species.keys())  # get list of all species available.
            # If no slice is given, load all configurations.
            if select_slice is None:
                select_slice = np.s_[:]  # set the numpy slice object.

        path_list = []
        for item in species:
            path_list.append(join_path(item, identifier))
        return database.load_data(path_list=path_list, select_slice=select_slice)

    @property
    def results(self):
        """
        Property to get access to the results in a dictionary

        Returns
        -------
        self._results: dict
            the actual dictionary with the results
        """
        return self._results

    @results.getter
    def results(self):
        """
        Getter to retrieve the results from the YAML file in a dictionary
        :return: dict

        Returns
        -------
        self._results: dict
            the actual dictionary with the results from the YAML file
        """

        with open(os.path.join(self.database_path, 'system_properties.yaml')) as pfr:
            self._results = yaml.load(pfr, Loader=yaml.Loader)  # collect the data in the yaml file

        return self._results

    @results.setter
    def results(self, result_dict):
        """
        Setter to dump the results to the YAML file

        Parameters
        ----------
        result_dict: dict
            dictionary with the results. It will store them in the system_properties.yaml file.

        """

        with open(os.path.join(self.database_path, 'system_properties.yaml'), 'w') as pfw:
            yaml.dump(result_dict, pfw)

    def write_xyz(self, dump_property: str = "Positions", species: list = None, atom_select: np.s_ = np.s_[:],
                  time_slice: np.s_ = np.s_[:], name: str = 'dump.xyz'):
        """
        Write an xyz file from a database dataset
        """
        if species is None:
            species = list(self.species)

        database = Database(name=os.path.join(self.database_path, "database.hdf5"), architecture='simulation')

        path_list = [join_path(s, dump_property) for s in species]
        if len(species) == 1:
            data_matrix = [database.load_data(path_list=path_list, select_slice=np.s_[:, 0:500])]
        else:
            data_matrix = database.load_data(path_list=path_list, select_slice=np.s_[:, 0:500])
        n_atoms = sum(len(self.species[s]['indices']) for s in species)

        with open(f"{name}.xyz", 'w') as f:
            for i in tqdm(range(500), ncols=70):
                f.write(f"{n_atoms}\n")
                f.write("Generated by the mdsuite xyz writer\n")
                for j, atom_species in enumerate(species):
                    for atom in data_matrix[j].numpy():
                        f.write(
                            f"{atom_species:<2}    {atom[i][0]:>9.4f}    {atom[i][1]:>9.4f}    {atom[i][2]:>9.4f}\n")

    def export_data(self, group: str, key: str = None, sub_key: str = None):
        """
        Export data from the analysis database.

        Parameters
        ----------
        group : str
                Group in the database from which data should be loaded
        key  : str
                Additional identifier.
        sub_key : str
                Additional identifier
        Returns
        -------
        saves a csv to the working directory.
        """
        database = Database(name=os.path.join(self.database_path, 'analysis_data.hdf5'), architecture='analysis')
        database.export_csv(group=group, key=key, sub_key=sub_key)

    def summarise(self):
        """
        Summarise the properties of the experiment.
        """
        database = Database(name=os.path.join(self.database_path, 'database.hdf5'))
        print(f"MDSuite {self.analysis_name} Summary\n")
        print("==================================================================================\n")
        print(f"Name: {self.analysis_name}\n")
        print(f"Temperature: {self.temperature} K\n")
        print(f"Number of Configurations: {self.number_of_configurations}\n")
        print(f"Number of Atoms: {self.number_of_atoms}\n")
        print("Species Summary\n")
        print("---------------\n")
        print("Atomic Species\n")
        print("***************\n")
        for item in self.species:
            try:
                print(f"{item}: {len(self.species[item]['indices'])}\n")
            except ValueError:
                pass
        print("Molecule Species\n")
        print("*****************\n")
        for item in self.molecules:
            try:
                print(f"{item}: {len(self.molecules[item]['indices'])}\n")
            except ValueError:
                pass
        print("Database Information\n")
        print("---------------\n")
        print(f"Database Path: {self.database_path}/database.hdf5\n")
        print(f"Database Size: {os.path.getsize(os.path.join(self.database_path, 'database.hdf5'))*1e-9: 6.3f}GB\n")
        print(f"Data Groups: {database.get_database_summary()}\n")
        print("==================================================================================\n")
