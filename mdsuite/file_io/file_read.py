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
    mass: float = 0
    charge: float = 0


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
    box_l: list = None
    sample_step: float = None
    temperature: float = None
    simulation_data: dict = dataclasses.field(default_factory=dict)


class FileProcessor:
    """
    Class to handle reading from trajectory files.
    Output is supposed to be used by the experiment class for building and populating the trajectory database.
    """

    def get_metadata(self) -> TrajectoryMetadata:
        """
        Return the metadata required to build a database.
        """
        raise NotImplementedError('File Processors must implement metadata extraction')

    def get_next_n_configurations(self, n_configs: int) -> dict:
        """
        Read n_configs new configurations starting from the previous state.
        Parameters
        ----------
        n_configs: int
            Number of configurations to return

        Returns
        -------
        trajectory_chunk: dict
            A dict of the form {'species_1':{'Property_1':data_array_1, 'Property_2':data_array_2, ...}
                                'species_2': ...}
            The names of the species must coincide with the names given in TrajectoryMetadata.species_list.
            The names of the properties must coincide with the names given in the property list of each species.
            The data arrays must have the shape
            (SpeciesInfo.n_particles, n_configs, PropertyInfo.n_dims) for each species and property.

        """
        raise NotImplementedError('File Processors must implement data loading')


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
