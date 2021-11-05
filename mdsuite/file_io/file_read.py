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
import typing

from mdsuite.database.simulation_database import TrajectoryChunkData, TrajectoryMetadata


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


def skip_n_lines(file, n_lines: int):
    for _ in range(n_lines):
        next(file)


def read_n_lines(file, n_lines: int, start_at:int=None) -> list:
    """
    Get n_lines lines, starting at line number start_at.
    If start_at is None, read from the current file state
    Returns
    -------
    A list of strings, one string for each line
    """
    if start_at is not None:
        file.seek(0)
        skip_n_lines(file, start_at)
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
