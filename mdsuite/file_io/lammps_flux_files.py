"""
Module for reading lammps trajectory files

Summary
-------
"""

import numpy as np

from mdsuite.file_io.flux_files import FluxFile
# from .file_io_dict import lammps_flux
from mdsuite.utils.meta_functions import optimize_batch_size, join_path

var_names = {
    "Temperature": ["temp"],
    "Time": ["time"],
    "Thermal_Flux": ['c_flux_thermal[1]', 'c_flux_thermal[2]', 'c_flux_thermal[3]'],
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

    def __init__(self, obj, header_lines=9, file_path=None, sort: bool = False):
        """
        Python class constructor
        """

        super().__init__(obj, header_lines, file_path, sort=sort)  # fill the experiment class
        self.experiment.flux = True

    @staticmethod
    def _build_architecture(property_groups: dict, number_of_atoms: int,
                            number_of_configurations: int):
        """
        Build the database_path architecture for use by the database_path class

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
        architecture = {}  # instantiate the database_path architecture dictionary
        for observable in property_groups:
            architecture[f"{observable}/{observable}"] = (number_of_configurations, len(property_groups[observable]))
        return architecture

    def _get_line_length(self):
        """
        Get the length of a line of tensor_values in the file.

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
                updated with new information. This is necessary on the first run of tensor_values addition to the database_path. After
                this point, when new tensor_values is added, this is no longer required as other methods will take care of
                updating the properties that change with new tensor_values. In fact, it will set the number of configurations to
                only the new tensor_values, which will be wrong.
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
        self.experiment.property_groups = self._extract_properties(var_names, column_dict_properties)

        batch_size = optimize_batch_size(self.file_path, number_of_configurations)

        # get time related properties of the experiment
        with open(self.file_path) as f:
            # skip the header
            for _ in range(n_lines_header):
                next(f)
            time_0_line = f.readline().split()
            time_0 = float(time_0_line[column_dict_properties['time']])
            time_1_line = f.readline().split()
            time_1 = float(time_1_line[column_dict_properties['time']])

        sample_rate = (time_1 - time_0) / self.experiment.time_step

        # Update class attributes with calculated tensor_values
        self.experiment.batch_size = batch_size
        # self.properties = properties_summary
        self.experiment.number_of_configurations = number_of_configurations
        self.experiment.sample_rate = sample_rate
        self.time_0 = time_0

        # Get the number of atoms if not set in initialization
        if self.experiment.number_of_atoms is None:
            self.experiment.number_of_atoms = int(header[2][1])  # hopefully always in the same position

        # Get the volume, if not set in initialization
        if self.experiment.volume is None:
            print(float(header[4][7]))
            self.experiment.volume = float(header[4][7])  # hopefully always in the same position

        self.experiment.species = {'1': []}

        if update_class:
            self.experiment.batch_size = batch_size
            self.experiment.volume = self.experiment.volume

        else:
            self.experiment.batch_size = batch_size

        line_length = self._get_line_length()
        return self._build_architecture(self.experiment.property_groups,
                                        self.experiment.number_of_atoms,
                                        number_of_configurations), line_length

    def build_file_structure(self):
        """
        Build a skeleton of the file so that the database_path class can process it correctly.
        """

        structure = {}  # define initial dictionary

        for observable in self.experiment.property_groups:
            path = join_path(observable, observable)
            columns = self.experiment.property_groups[observable]
            structure[path] = {'indices': np.s_[:], 'columns': columns, 'length': 1}

        return structure
