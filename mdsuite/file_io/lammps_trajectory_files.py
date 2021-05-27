"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.
"""

"""
Module for reading lammps trajectory files

Summary
-------
"""

import sys

from mdsuite.file_io.trajectory_files import TrajectoryFile
from mdsuite.utils.exceptions import *
from mdsuite.utils.meta_functions import get_dimensionality
from mdsuite.utils.meta_functions import line_counter
from mdsuite.utils.meta_functions import optimize_batch_size
import copy

var_names = {
    "Positions": ['x', 'y', 'z'],
    "Scaled_Positions": ['xs', 'ys', 'zs'],
    "Unwrapped_Positions": ['xu', 'yu', 'zu'],
    "Scaled_Unwrapped_Positions": ['xsu', 'ysu', 'zsu'],
    "Velocities": ['vx', 'vy', 'vz'],
    "Forces": ['fx', 'fy', 'fz'],
    "Box_Images": ['ix', 'iy', 'iz'],
    "Dipole_Orientation_Magnitude": ['mux', 'muy', 'muz'],
    "Angular_Velocity_Spherical": ['omegax', 'omegay', 'omegaz'],
    "Angular_Velocity_Non_Spherical": ['angmomx', 'angmomy', 'angmomz'],
    "Torque": ['tqx', 'tqy', 'tqz'],
    "Charge": ['q'],
    "KE": ["c_KE"],
    "PE": ["c_PE"],
    "Stress": ['c_Stress[1]', 'c_Stress[2]', 'c_Stress[3]', 'c_Stress[4]', 'c_Stress[5]', 'c_Stress[6]']
}


class LAMMPSTrajectoryFile(TrajectoryFile):
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

    def __init__(self, obj, header_lines=9, file_path=None, sort: bool = False):
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

        return int(header[3][0])

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

    def _get_time_information(self, number_of_atoms: int):
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
        time_0 = float(header[1][0])  # Time in first configuration
        header = self._read_header(self.f_object, offset=number_of_atoms)  # get the second header
        time_1 = float(header[1][0])  # Time in second configuration
        self.f_object.seek(0)  # return to first line in file

        return time_1 - time_0

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

        id_index = header[8].index('id') - 2

        # Look for the element keyword
        try:
            # Look for element keyword in trajectory.
            if "element" in header[8]:
                element_index = header[8].index("element") - 2

            # Look for type keyword if element is not present.
            elif "type" in header[8]:
                element_index = header[8].index('type') - 2

            # Raise an error if no identifying keywords are found.
            else:
                raise NoElementInDump
        except NoElementInDump:
            print("Insufficient species or type identification available.")
            sys.exit(1)

        column_dict_properties = self._get_column_properties(header[8], skip_words=2)  # get properties

        property_groups = self._extract_properties(copy.deepcopy(var_names), column_dict_properties)

        box = [(float(header[5][1]) - float(header[5][0])),
               (float(header[6][1]) - float(header[6][0])),
               (float(header[7][1]) - float(header[7][0]))]

        # Loop over atoms in first configuration.
        for i in range(number_of_atoms):
            line = self.f_object.readline().split()
            line_length = len(line)
            if line[element_index] not in species_summary:
                species_summary[line[element_index]] = {}
                species_summary[line[element_index]]['indices'] = []

            # Update the index of the atom in the summary.
            if self.sort:
                species_summary[line[element_index]]['indices'].append(int(line[id_index]))
            else:
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
                updated with new information. This is necessary on the first run of tensor_values addition to the database_path. After
                this point, when new tensor_values is added, this is no longer required as other methods will take care of
                updating the properties that change with new tensor_values. In fact, it will set the number of configurations to
                only the new tensor_values, which will be wrong.

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
            self.experiment.number_of_configurations += number_of_configurations

        return self._build_architecture(species_summary, property_groups, number_of_configurations), line_length
