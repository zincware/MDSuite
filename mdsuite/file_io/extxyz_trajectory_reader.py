"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Module for reading extxyz trajectory files

Summary
-------
"""
import logging
from typing import Union, List, Dict, Tuple
import numpy as np
from mdsuite.file_io.trajectory_files import TrajectoryFile
from mdsuite.utils.meta_functions import get_dimensionality
from mdsuite.utils.meta_functions import line_counter
from mdsuite.utils.meta_functions import optimize_batch_size

log = logging.getLogger(__name__)

var_names = {
    "Positions": 'pos',
    "Velocities": 'vel',
    "Forces": 'forces',
    "Stress": 'stress',
    "PE": 'energies',
    "Time": 'time',
    "Lattice": 'Lattice',
    "Momenta": 'momenta'
}


class EXTXYZFileReader(TrajectoryFile):
    """
    Child class for the lammps file reader.

    Attributes
    ----------
    obj : object
            Experiment class instance to add to

    header_lines : int
            Number of header lines in the file format (lammps = 9)

    file_path : str
            Path to the trajectory file.
    """

    def __init__(self, obj, header_lines=2, file_path=None, sort: bool = False):
        """
        Python class constructor
        """

        super().__init__(obj, header_lines, file_path, sort=sort)  # fill the experiment class

        self.f_object = open(self.file_path)  # file object

    def _get_number_of_atoms(self):
        """
        Get the number of atoms

        Returns
        -------

        # user custom names for variables.
        if rename_cols is not None:
            var_names.update(rename_cols)

        """

        header = self._read_header(self.f_object)  # get the first header tensor_values
        self.f_object.seek(0)  # go back to the start of the file

        return int(header[0][0])

    def _get_number_of_configurations(self, number_of_atoms: int):
        """
        Get the number of configurations

        Parameters
        ----------
        number_of_atoms : int
                Number of atoms in each of the trajectories
        Returns
        -------

        """
        number_of_lines = line_counter(self.file_path)  # get the number of lines in the file

        return int(number_of_lines / (number_of_atoms + self.header_lines))

    @staticmethod
    def _get_time_value(data: list):
        """
        Extract the time value from the header.

        Parameters
        ----------
        data : list
                Header data to analyze.
        Returns
        -------
        time : Union[float, None]
                The time value.
        """
        time = None
        for item in data:
            if var_names['Time'] in item:
                try:
                    time = float(item.split('=')[-1])
                except ValueError:
                    time = float(item.split('=')[-1].split(',')[0])
        return time

    def _get_time_information(self, number_of_atoms: int) -> Union[float, None]:
        """
        Get time information.

        Parameters
        ----------
        number_of_atoms : int
                Number of atoms in each trajectory

        Returns
        -------

        """
        header = self._read_header(self.f_object)  # get the first header
        time_0 = self._get_time_value(header[1])
        header = self._read_header(self.f_object, offset=number_of_atoms)  # get the second header
        time_1 = self._get_time_value(header[1])  # Time in second configuration
        self.f_object.seek(0)  # return to first line in file

        if time_1 is not None:
            time = time_1 - time_0
        else:
            time = None

        return time

    @staticmethod
    def _read_lattice(data: list) -> Union[list, None]:
        """
        Get the lattice information
        Parameters
        ----------
        data : list
                header file to read

        Returns
        -------

        """
        lattice = None
        start = None
        for idx, item in enumerate(data):
            if var_names['Lattice'] in item:
                start = idx
                break

        if start is not None:
            for idx, item in enumerate(data[start:]):
                if item[-1] == '"':
                    stop = idx
                    break

        if stop is not None:
            lattice = data[start:stop + 1]
            lattice[0] = lattice[0].split('=')[1].replace('"', '')
            lattice[-1] = lattice[-1].replace('"', '')
            lattice = np.array(lattice).astype(float)

        return [lattice[0], lattice[4], lattice[8]]

    @staticmethod
    def _get_property_summary(data: list) -> Tuple[int, Dict[str, List[int]]]:
        """
        Get the property summary from the header data.

        Parameters
        ----------
        data : list
                Data to analyze
        Returns
        -------
        property_summary : dict
                A dictionary of properties and their location in the data file.
        """
        key_list = list(var_names.keys())
        value_list = list(var_names.values())
        for idx, item in enumerate(data):
            if 'Properties' in item:
                start = idx
        data = data[start].split('=')[1].replace(':S', '').replace(':R', '').split(':')
        index = 0
        property_summary = {}
        for idx, item in enumerate(data):
            if item == 'species':
                species_index = index
                index += 1
            if item in value_list:
                key = key_list[value_list.index(item)]
                length = int(data[int(idx + 1)])
                property_summary[key] = [index + i for i in range(length)]
                index += length

        return species_index, property_summary

    def _split_extxyz_properties(self, header: list) -> tuple:
        """
        Take in the extxyz property lines and get all necessary information.

        Parameters
        ----------
        header : list
                Header lines to read.
        Returns
        -------
        property_groups : dict
                Property groups and their position in the data.
        element_index : int
                Index at which elements are stored
        lattice : list
                Lattice parameters of the system.
        """
        lattice = self._read_lattice(header[1])
        species_index, property_summary = self._get_property_summary(header[1])

        return lattice, species_index, property_summary

    def _get_species_information(self, number_of_atoms: int):
        """
        Get the initial species information

        Parameters
        ----------
        number_of_atoms : int
                Number of atoms in each configuration
        """
        line_length: int = 0
        species_summary = {}  # instantiate the species summary
        header = self._read_header(self.f_object)  # get the header tensor_values

        box, element_index, property_groups = self._split_extxyz_properties(header)

        # Loop over atoms in first configuration.
        for i in range(number_of_atoms):
            line = self.f_object.readline().split()
            line_length = len(line)
            if line[element_index] not in species_summary:
                species_summary[line[element_index]] = {}
                species_summary[line[element_index]]['indices'] = []

            species_summary[line[element_index]]['indices'].append(i + self.header_lines)

        return species_summary, box, property_groups, line_length

    def process_trajectory_file(self, update_class: bool = True, rename_cols: dict = None):
        """
        Get additional information from the trajectory file

        In this method, there are several doc string styled comments. This is included as there are several components
        of the method that are all related to the analysis of the trajectory file.

        Parameters
        ----------
        rename_cols : dict
                Will map some observable to keys found in the dump file.
        update_class : bool
                Boolean decision on whether or not to update the class. If yes, the full saved class instance will be
                updated with new information. This is necessary on the first run of tensor_values addition to the
                database_path. After this point, when new tensor_values is added, this is no longer required as other
                methods will take care of updating the properties that change with new tensor_values. In fact, it will
                set the number of configurations to only the new tensor_values, which will be wrong.

        Returns
        -------
        architecture : dict
                Database architecture to be used by the class to build a new database_path.
        """

        # user custom names for variables.
        if rename_cols is not None:
            var_names.update(rename_cols)
        number_of_atoms = self._get_number_of_atoms()  # get the number of atoms
        number_of_configurations = self._get_number_of_configurations(number_of_atoms)  # get number of configurations
        sample_rate = self._get_time_information(number_of_atoms)  # get the sample rate
        batch_size = optimize_batch_size(self.file_path, number_of_configurations)  # get the batch size
        species_summary, box, property_groups, line_length = self._get_species_information(number_of_atoms)

        if update_class:
            log.debug("Updating class")
            self.experiment.batch_size = batch_size
            self.experiment.dimensions = get_dimensionality(box)
            self.experiment.box_array = box
            self.experiment.volume = box[0] * box[1] * box[2]
            self.experiment.species = species_summary
            self.experiment.number_of_atoms = number_of_atoms
            self.experiment.number_of_configurations += number_of_configurations
            self.experiment.sample_rate = sample_rate
            self.experiment.property_groups = property_groups

        else:
            self.experiment.batch_size = batch_size

        return self._build_architecture(species_summary, property_groups, number_of_configurations), line_length
