"""
Module for reading lammps trajectory files

Summary
-------
"""

import numpy as np
import os

from mdsuite.file_io.flux_files import FluxFile
# from .file_io_dict import lammps_flux
from mdsuite.utils.meta_functions import optimize_batch_size, join_path

from pathlib import Path

var_names = {
    "Temperature": ["temp"],
    "Time": ["time"],
    "Flux_Thermal": ['c_flux_thermal[1]', 'c_flux_thermal[2]', 'c_flux_thermal[3]'],
    "Stress_visc": ['pxy', 'pxz', 'pyz'],
}


class LAMMPSFluxFile(FluxFile):
    """
    Child class for the lammps file reader to read Flux files from LAMMPS.

    Attributes
    ----------
    obj : object
            Experiment class instance to add to

    header_lines : int
            Number of header lines in the file format (lammps = 9)

    file_path : str
            Path to the trajectory file.
    """

    def __init__(self, obj, header_lines=9, file_path=None):
        """
        Python class constructor
        """

        super().__init__(obj, header_lines, file_path)  # fill the parent class
        self.project.volume = None
        self.project.number_of_atoms = None
        self.project.flux = True

    @staticmethod
    def _build_architecture(property_groups: dict, number_of_atoms: int,
                            number_of_configurations: int):
        """
        Build the database architecture for use by the database class

        Parameters
        ----------
        species_summary : dict
                Species summary passed to the experiment class
        property_groups : dict
                Property information passed to the experiment class
        number_of_atoms : int
                Number of atoms in each configurations
        number_of_configurations : int
                Number of configurations in the file

        """
        architecture = {}  # instantiate the database architecture dictionary
        for observable in property_groups:
            architecture[f"{observable}/{observable}"] = (number_of_configurations, len(property_groups[observable]))
        return architecture

    def _get_line_length(self):
        """
        Get the length of a line of data in the file.

        Returns
        -------

        """
        with open(self.file_path) as f:
            for i in range(self.header_lines):
                f.readline()

            line_length = len(f.readline().split())

        return line_length

    def process_trajectory_file(self, update_class=True, rename_cols=None):
        """ Get additional information from the trajectory file

        In this method, there are several doc string styled comments. This is included as there are several components
        of the method that are all related to the analysis of the trajectory file.

        Parameters
        ----------
        rename_cols : dict
                Will map some observable to keys found in the dump file.
        update_class : bool
                Boolean decision on whether or not to update the class. If yes, the full saved class instance will be
                updated with new information. This is necessary on the first run of data addition to the database. After
                this point, when new data is added, this is no longer required as other methods will take care of
                updating the properties that change with new data. In fact, it will set the number of configurations to
                only the new data, which will be wrong.
        """

        # user custom names for variables.
        if rename_cols is not None:
            var_names.update(rename_cols)

        n_lines_header = 0  # number of lines of header
        with open(self.file_path) as f:
            header = []
            for line in f:
                n_lines_header += 1
                if line.startswith("#"):
                    header.append(line.split())
                else:
                    header_line = line.split()  # after the comments, we have the line with the variables
                    break

        self.header_lines = n_lines_header

        with open(self.file_path) as f:
            number_of_configurations = sum(1 for _ in f) - n_lines_header

        # Find properties available for analysis
        column_dict_properties = self._get_column_properties(header_line)
        self.project.property_groups = self._extract_properties(var_names, column_dict_properties)

        batch_size = optimize_batch_size(self.file_path, number_of_configurations)

        # get time related properties of the system
        with open(self.file_path) as f:
            # skip the header
            for _ in range(n_lines_header):
                next(f)
            time_0_line = f.readline().split()
            time_0 = float(time_0_line[column_dict_properties['time']])
            time_1_line = f.readline().split()
            time_1 = float(time_1_line[column_dict_properties['time']])

        sample_rate = (time_1 - time_0) / self.project.time_step
        time_n = (number_of_configurations - number_of_configurations % batch_size) * sample_rate

        # Update class attributes with calculated data
        self.project.batch_size = batch_size
        # self.properties = properties_summary
        self.project.number_of_configurations = number_of_configurations
        self.project.sample_rate = sample_rate
        self.time_0 = time_0

        # Get the number of atoms if not set in initialization
        if self.project.number_of_atoms is None:
            self.project.number_of_atoms = int(header[2][1])  # hopefully always in the same position

        # Get the volume, if not set in initialization
        if self.project.volume is None:
            self.project.volume = float(header[4][7])  # hopefully always in the same position

        self.project.species = {'1': []}

        if update_class:
            self.project.batch_size = batch_size

        else:
            self.project.batch_size = batch_size
            # return [1, 1, 1, number_of_configurations]

        line_length = self._get_line_length()
        return self._build_architecture(self.project.property_groups,
                                        self.project.number_of_atoms,
                                        number_of_configurations), line_length

    def build_file_structure(self):
        """
        Build a skeleton of the file so that the database class can process it correctly.
        """

        structure = {}  # define initial dictionary

        for observable in self.project.property_groups:
            path = join_path(observable, observable)
            columns = self.project.property_groups[observable]
            structure[path] = {'indices': np.s_[:], 'columns': columns, 'length': 1}

        return structure
