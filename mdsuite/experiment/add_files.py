"""
MDSuite: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
"""
import logging

from mdsuite.database.simulation_database import Database
from mdsuite.file_io.file_io_dict import dict_file_io
from mdsuite.file_io.file_read import FileProcessor
from mdsuite.utils.exceptions import ElementMassAssignedZero
from mdsuite.utils.meta_functions import join_path

from tqdm import tqdm
import pathlib
import pubchempy as pcp
import importlib.resources
import json
import numpy as np
import copy

log = logging.getLogger(__name__)


def _load_trajectory_reader(file_format, trajectory_file, sort: bool = False):
    # TODO replace by proper instantiation
    try:
        class_file_io, file_type = dict_file_io[
            file_format
        ]  # file type is per atoms or flux.
    except KeyError:
        raise ValueError(
            f"{file_format} not found! \n"
            f"Available io formats are are: {[key for key in dict_file_io.keys()]}"
        )
    return class_file_io(file_path=trajectory_file, sort=sort), file_type


def _species_list_to_architecture_dict(species_list, n_configurations):
    # TODO let the database handler use the species list directly instead of the dict
    architecture = {}
    for sp_info in species_list:
        architecture[sp_info.name] = {}
        for prop_info in sp_info.properties:
            architecture[sp_info.name][prop_info.name] = (
                sp_info.n_particles,
                n_configurations,
                prop_info.n_dim)
    return architecture


class ExperimentAddingFiles:
    """Parent class that handles the file additions"""

    def __init__(self):
        """Constructor of the ExperimentAddingFiles class"""
        self.database_path: str = ""
        self.n_configurations_stored = 0
        self.memory_requirements = None

        self.species_list = None
        self.read_files = None
        self.version = None

    def add_data(
            self,
            trajectory_file: str = None,
            file_format: str = "lammps_traj",
            force: bool = False,
            update_with_pubchempy = True
    ):
        """
        Add tensor_values to the database_path

        Parameters
        ----------
        file_format :
                Format of the file being read in. Default is file_path
        trajectory_file : str
                Trajectory file to be process and added to the database_path.
        force : bool
                If true, a file will be read regardless of if it has already
                been seen.
        """

        # Check if there is a trajectory file.
        if trajectory_file is None:
            raise ValueError("You need to pass a trajectory_file for data loading")

        # make absolute path
        traj_file_path = pathlib.Path(trajectory_file)
        traj_file_path.resolve()

        already_read = traj_file_path in self.read_files
        if already_read and not force:
            log.info(
                "This file has already been read, skipping this now."
                "If this is not desired, please add force=True "
                "to the command."
            )
            return  # End the method.

        # todo trajectory_reader should be an argument of this function
        trajectory_reader, file_type = _load_trajectory_reader(
            file_format, trajectory_file
        )
        database = Database(
            name=pathlib.Path(self.database_path, "database.hdf5").as_posix(),
        )

        # Check to see if a database_path exists
        # todo make method of database class
        database_path = pathlib.Path(
            self.database_path, "database.hdf5"
        )  # get theoretical path.

        metadata = trajectory_reader.get_metadata()
        architecture = _species_list_to_architecture_dict(metadata.species_list,
                                                          metadata.n_configurations)
        if not database_path.exists():
            # todo store the other metadata
            database.initialize_database(architecture)
        else:
            database.resize_datasets(architecture)

        for i, batch in enumerate(trajectory_reader.get_configurations_generator()):
            database.add_data(
                chunk=batch,
                start_idx=self.n_configurations_stored
            )
            self.n_configurations_stored += batch.chunk_size

        self.version += 1

        self.species_list = copy.deepcopy(metadata.species_list)
        if update_with_pubchempy:
            update_species_attributes_with_pubchempy(self.species_list)

        self.memory_requirements = database.get_memory_information()

        # set at the end, because if something fails, the file was not properly read.
        self.read_files = self.read_files + [traj_file_path]

    def load_matrix(
            self,
            property_name: str = None,
            species: list = None,
            select_slice: np.s_ = None,
            path: list = None,
    ):
        """
        Load a desired property matrix.

        Parameters
        ----------
        property_name : str
                Name of the matrix to be loaded, e.g. 'Unwrapped_Positions', 'Velocities'
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
        database = Database(name=pathlib.Path(self.database_path, "database.hdf5").as_posix())

        if path is not None:
            return database.load_data(path_list=path, select_slice=select_slice)

        else:
            # If no species list is given, use all species in the Experiment.
            if species is None:
                species = [sp.name for sp in self.species_list]
            # If no slice is given, load all configurations.
            if select_slice is None:
                select_slice = np.s_[:]  # set the numpy slice object.

        path_list = []
        for item in species:
            path_list.append(join_path(item, property_name))
        return database.load_data(path_list=path_list, select_slice=select_slice)


def update_species_attributes_with_pubchempy(species_list):
    """
    Add information to the species dictionary

    A fundamental part of this package is species specific analysis. Therefore, the
    Pubchempy package is used to add important species specific information to the
    class. This will include the charge of the ions which will be used in
    conductivity calculations.

    """
    with importlib.resources.open_text(
            "mdsuite.data", "PubChemElements_all.json"
    ) as json_file:
        pse = json.loads(json_file.read())

    # Try to get the species tensor_values from the Periodic System of Elements file

    # set/find masses
    for sp_info in species_list:
        for entry in pse:
            if pse[entry][1] == sp_info.name:
                sp_info.mass = [float(pse[entry][3])]

                # If gathering the tensor_values from the PSE file was not successful
    # try to get it from Pubchem via pubchempy
    for sp_info in species_list:
        if sp_info.mass is None:
            try:
                temp = pcp.get_compounds(sp_info.name, "name")
                temp[0].to_dict(
                    properties=[
                        "atoms",
                        "bonds",
                        "exact_mass",
                        "molecular_weight",
                        "elements",
                    ]
                )
                sp_info.mass = [temp[0].molecular_weight]
                log.debug(temp[0].exact_mass)
            except (ElementMassAssignedZero, IndexError):
                sp_info.mass = 0.0
                log.warning(f"WARNING element {sp_info.name} has been assigned mass=0.0")
    return species_list
