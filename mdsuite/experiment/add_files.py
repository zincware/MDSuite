"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Class to handle file additions
"""
import logging

from mdsuite.database.simulation_database import Database
from mdsuite.file_io.file_io_dict import dict_file_io
from mdsuite.file_io.file_read import FileProcessor
from mdsuite.utils.exceptions import ElementMassAssignedZero
from mdsuite.utils.meta_functions import join_path

from tqdm import tqdm
from pathlib import Path
import pubchempy as pcp
import importlib.resources
import json
import numpy as np

log = logging.getLogger(__name__)


class ExperimentAddingFiles:
    """Parent class that handles the file additions"""

    def __init__(self):
        self.database_path: str = ""
        self.memory_requirements = None
        self.number_of_configurations = None
        self.batch_size = None
        self.number_of_atoms = None
        self.species = None
        self.property_groups = None
        self.read_files = None

    def add_data(self,
                 trajectory_file: str = None,
                 file_format: str = 'lammps_traj',
                 rename_cols: dict = None,
                 sort: bool = False,
                 force: bool = False):
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
        force : bool
                If true, a file will be read regardless of if it has already
                been seen.
        """

        # Check if there is a trajectory file.
        if trajectory_file is None:
            raise ValueError("No tensor_values has been given")

        file_check = self._check_read_files(trajectory_file)
        if file_check:
            if force:
                pass
            else:
                log.info('This file has already been read, skipping this now.'
                         'If this is not desired, please add force=True '
                         'to the command.')
                return  # End the method.

        # Load the file reader and the database_path object
        trajectory_reader, file_type = self._load_trajectory_reader(file_format, trajectory_file, sort=sort)
        database = Database(name=Path(self.database_path, "database.hdf5").as_posix(), architecture='simulation')

        # Check to see if a database_path exists
        database_path = Path(self.database_path, 'database.hdf5')  # get theoretical path.

        if file_type == 'flux':
            flux = True
        else:
            flux = False

        if database_path.exists():
            self._update_database(trajectory_reader,
                                  trajectory_file,
                                  database,
                                  rename_cols,
                                  sort=sort)
        else:
            self._build_new_database(trajectory_reader,
                                     trajectory_file,
                                     database,
                                     rename_cols=rename_cols,
                                     flux=flux,
                                     sort=sort)

        self.build_species_dictionary()
        self.memory_requirements = database.get_memory_information()

    def _check_read_files(self, file_path: str):
        """
        Check if a file has been read before and add it to the hidden file.

        Parameters
        ----------
        file_path : str
                Path to the file.

        Returns
        -------

        """
        file_path = Path(file_path)
        if file_path in self.read_files:
            return True
        else:
            self.read_files = file_path
            return False

    def _load_trajectory_reader(self, file_format, trajectory_file, sort: bool = False):
        try:
            class_file_io, file_type = dict_file_io[file_format]  # file type is per atoms or flux.
        except KeyError:
            raise ValueError(f'{file_format} not found! \n'
                             f'Available io formats are are: {[key for key in dict_file_io.keys()]}')
        return class_file_io(self, file_path=trajectory_file, sort=sort), file_type

    def _update_database(self, trajectory_reader: FileProcessor, trajectory_file: str, database: Database,
                         rename_cols: dict, flux: bool = False, sort: bool = False):
        """
        Update the database rather than build a new database.

        Returns
        -------
        Updates the current database.
        """
        counter = self.number_of_configurations
        architecture, line_length = trajectory_reader.process_trajectory_file(rename_cols=rename_cols,
                                                                              update_class=False)
        number_of_new_configurations = self.number_of_configurations - counter
        database.resize_dataset(architecture)  # initialize the database_path
        batch_range = int(number_of_new_configurations / self.batch_size)  # calculate the batch range
        remainder = number_of_new_configurations - (batch_range * self.batch_size)
        structure = trajectory_reader.build_file_structure()  # build the file structure
        with open(trajectory_file, 'r') as f_object:
            for _ in tqdm(range(batch_range), ncols=70):
                database.add_data(data=trajectory_reader.read_configurations(self.batch_size, f_object, line_length),
                                  structure=structure,
                                  start_index=counter,
                                  batch_size=self.batch_size,
                                  flux=flux,
                                  n_atoms=self.number_of_atoms,
                                  sort=sort)
                counter += self.batch_size

            if remainder > 0:
                structure = trajectory_reader.build_file_structure(batch_size=remainder)  # build the file structure
                database.add_data(data=trajectory_reader.read_configurations(remainder, f_object, line_length),
                                  structure=structure,
                                  start_index=counter,
                                  batch_size=remainder,
                                  flux=flux)

    def _build_new_database(self, trajectory_reader: FileProcessor, trajectory_file: str, database: Database,
                            rename_cols: dict, flux: bool = False, sort: bool = False):
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

        with open(trajectory_file, 'r') as f_object:
            for _ in tqdm(range(batch_range), ncols=70):
                database.add_data(data=trajectory_reader.read_configurations(self.batch_size, f_object, line_length),
                                  structure=structure,
                                  start_index=counter,
                                  batch_size=self.batch_size,
                                  flux=flux,
                                  n_atoms=self.number_of_atoms,
                                  sort=sort)
                counter += self.batch_size

            if remainder > 0:
                structure = trajectory_reader.build_file_structure(batch_size=remainder)  # build the file structure
                database.add_data(data=trajectory_reader.read_configurations(remainder, f_object, line_length),
                                  structure=structure,
                                  start_index=counter,
                                  batch_size=remainder,
                                  flux=flux,
                                  n_atoms=self.number_of_atoms,
                                  sort=sort)

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
        species = dict(self.species)
        for element in self.species:
            species[element]['charge'] = [0.0]
            for entry in pse:
                if pse[entry][1] == element:
                    species[element]['mass'] = [float(pse[entry][3])]

        # If gathering the tensor_values from the PSE file was not successful try to get it from Pubchem via pubchempy
        for element in self.species:
            if 'mass' not in self.species[element]:
                try:
                    temp = pcp.get_compounds(element, 'name')
                    temp[0].to_dict(properties=['atoms', 'bonds', 'exact_mass', 'molecular_weight', 'elements'])
                    species[element]['mass'] = [temp[0].molecular_weight]
                    log.debug(temp[0].exact_mass)
                except (ElementMassAssignedZero, IndexError):
                    species[element]['mass'] = [0.0]
                    log.warning(f'WARNING element {element} has been assigned mass=0.0')
        # self.save_class()
        self.species = species

    def load_matrix(self, identifier: str = None, species: list = None, select_slice: np.s_ = None, path: list = None):
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
        database = Database(name=Path(self.database_path, 'database.hdf5').as_posix())

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
