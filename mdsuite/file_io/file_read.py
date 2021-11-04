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
import abc
import dataclasses
import numpy as np
import typing


@dataclasses.dataclass
class PropertyInfo:
    """
    Information of a trajectory property.
    example:
    pos_info = PropertyInfo('Positions', 3)
    vel_info = PropertyInfo('Velocities', 3)

    Attributes
    ----------
    name:
        The name of the property
    n_dims:
        The dimensionality of the property
    """
    name: str
    n_dims: int


@dataclasses.dataclass
class SpeciesInfo:
    """
    Information of a species

    Attributes
    ----------
    name
        Name of the species (e.g. 'Na')
    n_particles
        Number of particles of that species
    properties: list of PropertyInfo
        List of the properties that were recorded for the species
    mass and charge are optional
    """
    name: str
    n_particles: int
    properties: list
    mass: float = None
    charge: float = 0

    def __eq__(self, other):
        same = self.name == other.name and \
               self.n_particles == other.n_particles and \
               self.mass == other.mass and \
               self.charge == other.charge
        if len(self.properties) != len(other.properties):
            return False

        for prop_s, prop_o in zip(self.properties, other.properties):
            same = same and prop_s == prop_o
        return same


@dataclasses.dataclass
class TrajectoryMetadata:
    """
    This metadata must be extracted from trajectory files to build the database into which the trajectory will be stored

    Attributes
    ----------
    n_configurations:
        Number of configurations of the whole trajectory.
    species_list: list of SpeciesInfo
        The information about all species in the system.
    box_l: list of float
        The simulation box size in three dimensions
    sample_step: optional
        The time between consecutive configurations.
        E.g. for a simulation with time step 0.1 where the trajectory is written every 5 steps: sample_step = 0.5
        Does not have to be specified (e.g. configurations from Monte Carlo scheme), but is needed for all dynamic observables.
    temperatyre: optional
        The set temperature of the system.
        Optional because only applicable for MD simulations with thermostat. Needed for certain observables.
    simulation_data: optional
        All other simulation data that can be extracted from the trajectory metadata.
        E.g. software version, pressure in NPT simulations, time step, ...

    """
    n_configurations: int
    species_list: list
    box_l: list
    sample_step: float = None
    temperature: float = None
    simulation_data: dict = dataclasses.field(default_factory=dict)


class TrajectoryChunkData:
    """
    Class to specify the data format for transfer from the file to the database
    """
    def __init__(self, species_list: list, chunk_size: int):
        """

        Parameters
        ----------
        species_list:
            List of SpeciesInfo.
            It contains the information which species are there and which properties are recoreded for each
        chunk_size:
            The number of configurations to be stored in this chunk
        """
        self.chunk_size = chunk_size
        self.species_list = species_list
        self._data = dict()
        for sp_info in species_list:
            self._data[sp_info.name] = dict()
            for prop_info in sp_info.properties:
                self._data[sp_info.name][prop_info.name] = np.zeros((chunk_size, sp_info.n_particles, prop_info.n_dims))

    def add_data(self, data: np.ndarray, config_idx, species_name, property_name):
        """
        Add configuration data to the chunk
        Parameters
        ----------
        data:
            The data to be added, with shape (n_configs, n_particles, n_dims).
            n_particles and n_dims relates to the species and the property that is being added
        config_idx:
            Start index of the configs that are being added.
        species_name
            Name of the species to which the data belongs
        property_name
            Name of the property being added

        example:
        Storing velocity Information for 42 Na atoms in the 17th iteration of a loop that reads 5 configs per loop:
        add_data(vel_array, 16*5, 'Na', 'Velocities')
        where vel.data.shape == (5,42,3)

        """
        n_configs = len(data)
        self._data[species_name][property_name][config_idx:config_idx + n_configs, ...] = data

    def get_data(self):
        return self._data


class FileProcessor(abc.ABC):
    """
    Class to handle reading from trajectory files.
    Output is supposed to be used by the experiment class for building and populating the trajectory database.
    """
    def __str__(self):
        """
        Return a unique string representing this FileProcessor. (The absolute file path, for example)
        """
        raise NotImplementedError('File Processors must implement a string')

    def get_metadata(self) -> TrajectoryMetadata:
        """
        Return the metadata required to build a database.
        """
        raise NotImplementedError('File Processors must implement metadata extraction')

    def get_configurations_generator(self) -> typing.Iterator[TrajectoryChunkData]:
        """
        Yield configurations. Batch size must be determined by the FileProcessor
        Parameters
        ----------

        Returns
        -------
        generator that yields TrajectoryChunkData
        """
        raise NotImplementedError('File Processors must implement data loading')


def assert_species_list_consistent(sp_list_0, sp_list_1):
    for sp_info_data, sp_info_mdata in zip(sp_list_0, sp_list_1):
        if sp_info_data != sp_info_mdata:
            raise ValueError('Species information from data and metadata are inconsistent')


def read_n_lines(file, n_lines: int, start_at=0) -> list:
    """
    Get n_lines lines, starting at start
    Returns
    -------
    A list of strings, one string for each line
    """
    file.seek(start_at)
    return [next(file) for _ in range(n_lines)]


def extract_properties_from_header(header_property_names: list,
                                   database_correspondence_dict: dict,
                                   ) -> dict:
    """
    Takes the property names from a file header, sees if there is a corresponding mdsuite property
    in database_correspondence_dict.
    Returns a dict that links the mdsuite property names to the column indices at which they can be found in the file.

    Parameters
    ----------
    header_property_names
        The names of the columns in the data file
    database_correspondence_dict
        The translation between mdsuite properties and the column names of the respective file format.
        lammps example:
        {"Positions": ["x", "y", "z"],
         "Unwrapped_Positions": ["xu", "yu", "zu"],

    Returns
    -------
    trajectory_properties : dict
        A dict of the form {'MDSuite_Property_1': [column_indices], 'MDSuite_Property_2': ...}
        Example {'Unwrapped_Positions': [2,3,4], 'Velocities': [5,6,7]}
    """

    column_dict_properties = {variable: idx for idx, variable in enumerate(header_property_names)}
    # for each property label (position, velocity,etc) in the lammps definition
    for property_label, property_names in database_correspondence_dict.items():
        # for each coordinate for a given property label (position: x, y, z),
        # get idx and the name
        for idx, property_name in enumerate(property_names):
            if property_name in column_dict_properties.keys():  # if this name (x) is in the input file properties
                # we change the lammps_properties_dict replacing the string of the
                # property name by the column name
                database_correspondence_dict[property_label][
                    idx
                ] = column_dict_properties[property_name]

    # trajectory_properties only need the labels with the integer columns, then we
    # only copy those
    trajectory_properties = {}
    for property_label, properties_columns in database_correspondence_dict.items():
        if all(
                [
                    isinstance(property_column, int)
                    for property_column in properties_columns
                ]
        ):
            trajectory_properties[property_label] = properties_columns

    return trajectory_properties
