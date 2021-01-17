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
from importlib.resources import open_text
from pathlib import Path

import h5py as hf
import numpy as np
import pubchempy as pcp
import tensorflow as tf
from diagrams import Diagram, Cluster
from diagrams.aws.compute import ECS
from diagrams.aws.database import RDS

from mdsuite import data as static_data
from mdsuite.calculators.computations_dict import dict_classes_computations
from mdsuite.file_io.file_io_dict import dict_file_io
from mdsuite.utils.units import units_dict


class Experiment():

    def __init__(self, analysis_name, storage_path='./', time_step=1.0, temperature=0, units='real'):
        """
        Initialise the experiment class.
        """

        # Taken upon instantiation
        self.analysis_name = analysis_name  # Name of the experiment.
        self.storage_path = storage_path  # Where to store the data - should have sufficient free space.
        self.temperature = temperature  # Temperature of the system.
        self.time_step = time_step  # Timestep chosen for the simulation.

        # Added from trajectory file
        self.units = self.units_to_si(units)  # Units used during the simulation.
        self.number_of_configurations = 0  # Number of configurations in the trajectory.

        # Memory properties
        self.memory_requirements = {}

        # Check if the experiment exists and load if it does.
        self._load_or_build()

        self.build_dictionary_results()

    def _load_or_build(self):
        test_dir = Path(f"{self.storage_path}/{self.analysis_name}")  # get the theoretical directory

        # Check if the experiment exists and load if it does.
        if test_dir.exists():
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

        save_file = open(f"{self.storage_path}/{self.analysis_name}/{self.analysis_name}.bin", 'wb')  # construct file
        save_file.write(pickle.dumps(self.__dict__))  # write to file
        save_file.close()

    @staticmethod
    def units_to_si(units_system):
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

        >>> units_to_si('metal')
        {'time': 1e-12, 'length': 1e-10, 'energy': 1.6022e-19}
        >>> units_to_si({'time': 1e-12, 'length': 1e-10, 'energy': 1.6022e-19, 'NkTV2p':1.6021765e6, 'boltzman':8.617343e-5})
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

    def _build_model(self):
        """
        Build the 'experiment' for the analysis

        A method to build the database in the hdf5 format. Within this method, several other are called to develop the
        database skeleton, get configurations, and process and store the configurations. The method is accompanied
        by a loading bar which should be customized to make it more interesting.
        """

        # Create new analysis directory and change into it
        try:
            os.mkdir(f'{self.storage_path}/{self.analysis_name}')  # Make the experiment directory
            os.mkdir(f'{self.storage_path}/{self.analysis_name}/Figures')  # Create a directory to save images
            os.mkdir(f'{self.storage_path}/{self.analysis_name}/data')  # Create a directory for data

        except FileExistsError:  # throw exception if the file exits
            return

        self.save_class()  # save the class state.

        print(f"** An experiment has been added entitled {self.analysis_name} **")

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

        database = hf.File("{0}/{1}/{1}.hdf5".format(self.filepath, self.analysis_name), "r")
        with Diagram("Web Service", show=True, direction='TB'):
            head = RDS("Database")  # set the head database object
            for item in database:
                with Cluster(f"{item}"):
                    group_list = []
                    for property_group in database[item]:
                        group_list.append(ECS(property_group))  # construct a list of equal level groups
                head >> group_list  # append these groups to the head object

    def add_data(self, trajectory_file=None, file_format='lammps_traj'):
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

        self.file_format = file_format
        self.trajectory_file = trajectory_file  # Update the current class trajectory file

        # Check which type of file it is: flux or per atom
        trajectory_reader, file_type = self._get_system_properties(file_format)

        self.file_type = file_type  # flux or per atom

        # Check to see if a database exists
        test_db = Path(f"{self.storage_path}/{self.analysis_name}/{self.analysis_name}.hdf5")  # get theoretical path.
        if test_db.exists():
            self._update_database(trajectory_reader)
        else:
            self._build_new_database(trajectory_reader)

        self.collect_memory_information()  # Update the memory information
        self.save_class()  # Update the class state.

    def _build_new_database(self, trajectory_reader):
        """
        Build a new database
        """
        trajectory_reader.process_trajectory_file()  # get properties of the trajectory and update the class
        trajectory_reader.build_database_skeleton()  # Build the database skeleton
        trajectory_reader.fill_database()  # Fill the database with trajectory data
        if self.file_type == 'traj':
            self.build_species_dictionary()  # Add data to the species dictionary.
        self.save_class()  # Update the class state

    def _get_system_properties(self, file_format):
        try:
            class_file_io, file_type = dict_file_io[file_format]  # file type is per atoms or flux.
        except KeyError:
            # TODO: maybe this exception can be done better, but I dont know enough about handling exceptions.
            print(f'{file_format} not found')
            print(f'Available io formats are are:')
            [print(key) for key in dict_file_io.keys()]
            sys.exit(1)
        return class_file_io(self), file_type

    def build_species_dictionary(self):
        """
        Add information to the species dictionary

        A fundamental part of this package is species specific analysis. Therefore, the mendeleev python package is
        used to add important species specific information to the class. This will include the charge of the ions which
        will be used in conductivity calculations.

        """
        with open_text(static_data, 'PubChemElements_all.json') as json_file:
            PSE = json.loads(json_file.read())

        # Try to get the species data from the Periodic System of Elements file
        for element in self.species:
            self.species[element]['charge'] = [0.0]
            for entry in PSE:
                if PSE[entry][1] == element:
                    self.species[element]['mass'] = [float(PSE[entry][3])]

        # If gathering the data from the PSE file was not successful try to get it from Pubchem via pubchempy
        for element in self.species:
            if not 'mass' in self.species[element]:
                try:
                    temp = pcp.get_compounds(element, 'name')
                    temp[0].to_dict(properties=['atoms', 'bonds', 'exact_mass', 'molecular_weight', 'elements'])
                    self.species[element]['mass'] = temp[0].molecular_weight
                    print(temp[0].exact_mass)
                except:
                    self.species[element]['mass'] = [0.0]
                    print(f'WARNING element {element} has been assigned mass=0.0')

    def collect_memory_information(self):
        """
        Get information about dataset memory requirements

        This method will simply get the size of all the datasets in the database such that efficient memory management
        can be performed during analysis.
        """

        with hf.File("{0}/{1}/{1}.hdf5".format(self.storage_path, self.analysis_name), "r+") as db:
            for item in self.species:  # Loop over the species keys
                self.memory_requirements[item] = {}  # construct a new dict entry
                for group, columns in self.property_groups.items():  # Loop over property groups
                    if len(columns) == 1:  # if it a scalar quantity
                        self.memory_requirements[item][group] = db[item][group].nbytes  # Update the dictionary.
                    else:
                        memory = 0  # Dummy variable for memory
                        for dataset in db[item][group]:  # Loop over the datasets in the group
                            memory += db[item][group][dataset].nbytes  # Sum over the memory for each dataset
                        self.memory_requirements[item][group] = memory  # Update the dictionary.

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
            return property_matrix  # return the full tensor object.

    def build_dictionary_results(self):
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
        self.kirkwood_buff_integral_state = True  # Set true if it has been calculated

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

    def dump_results_json(self):
        """
        Dump a json file.

        Returns
        -------

        """
        filename = Path(f"{self.storage_path}/{self.analysis_name}.json")
        with open(filename, 'w') as fp:
            json.dump(self.results, fp, indent=4, sort_keys=True)
