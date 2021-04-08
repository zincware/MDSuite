"""
Parent class for file processing

Summary
-------
"""

import abc
from typing import TextIO


class FileProcessor(metaclass=abc.ABCMeta):
    """
    Parent class for file reading and processing

    Attributes
    ----------
    obj, experiment : object

            File object to be opened and read in.
    header_lines : int
            Number of header lines in the file format being read.
    """

    def __init__(self, obj, header_lines, file_path):
        """
        Python constructor

        Parameters
        ----------
        obj : object
                Experiment class instance to add to.

        header_lines : int
                Number of header lines in the given file format.
        """

        self.experiment = obj  # Experiment class instance to add to.

        self.header_lines = header_lines  # Number of header lines in the given file format.
        self.file_path = file_path   # path to the file being read

    @abc.abstractmethod
    def process_trajectory_file(self, rename_cols: dict = None, update_class: bool = True):
        """
        Get property groups from the trajectory
        This method is dependent on the code that generated the file. So it should be implemented in a grandchild class.
        """

        return

    @abc.abstractmethod
    def build_file_structure(self, batch_size: int = None):
        """
        Build a skeleton of the file so that the database_path class can process it correctly.
        """

        return

    @abc.abstractmethod
    def read_configurations(self, number_of_configurations: int, file_object: TextIO, line_length: int):
        """
        Read in a number of configurations from a file

        Parameters
        ----------
        line_length : int
             Length of each line of tensor_values to be read in. Necessary for instantiation.
        number_of_configurations : int
                Number of configurations to be read in.
        file_object : experiment
                File object to be read from.

        Returns
        -------
        configuration tensor_values : np.array
                Data read in from the file object.
        """

        return

    @staticmethod
    def _extract_properties(database_correspondence_dict, column_dict_properties):
        """
        Construct generalized property array

        Takes the lammps properties dictionary and constructs an array of properties which can be used by the species
        class.

        Parameters
        ----------
        properties_dict : dict
                A dictionary of all the available properties in the trajectory. This dictionary is built only from the
                 LAMMPS symbols and therefore must be again processed to extract the useful information.

        Returns
        -------
        trajectory_properties : dict
                A dictionary of the keyword labelled properties in the trajectory. The  values of the dictionary keys
                correspond to the array location of the specific piece of tensor_values in the set.
        """

        # for each property label (position, velocity,etc) in the lammps definition
        for property_label, property_names in database_correspondence_dict.items():
            # for each coordinate for a given property label (position: x, y, z), get idx and the name
            for idx, property_name in enumerate(property_names):
                if property_name in column_dict_properties.keys():  # if this name (x) is in the input file properties
                    # we change the lammps_properties_dict replacing the string of the property name by the column name
                    database_correspondence_dict[property_label][idx] = column_dict_properties[property_name]

        # trajectory_properties only needs the labels with the integer columns, then we one copy those
        trajectory_properties = {}
        for property_label, properties_columns in database_correspondence_dict.items():
            if all([isinstance(property_column, int) for property_column in properties_columns]):
                trajectory_properties[property_label] = properties_columns

        return trajectory_properties

    @staticmethod
    def _get_column_properties(header_line, skip_words=0) -> dict:
        """
        Given a line of text with the header, split it, enumerate and put in a dictionary.
        This is used to create the column - variable correspondance (see self._extract_properties)

        Parameters
        ----------
                header_line: str
                        Header line to split
        Returns
        -------
                properties_summary : dict
        """
        header_line = header_line[skip_words:]
        properties_summary = {variable: idx for idx, variable in enumerate(header_line)}

        return properties_summary
