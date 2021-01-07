"""
Module for reading lammps trajectory files

Summary
-------
"""

from mdsuite.file_io.file_read import FileProcessor
from mdsuite.utils.constants import lammps_properties_dict
from mdsuite.utils.exceptions import *
from mdsuite.utils.meta_functions import line_counter
from mdsuite.utils.meta_functions import optimize_batch_size
from mdsuite.utils.meta_functions import get_dimensionality


class LAMMPSTrajectoryFile(FileProcessor):
    """
    Child class for the lammps file reader.

    Attributes
    ----------
    obj : object
            Experiment class instance to add to

    header_lines : int
            Number of header lines in the file format (lammps = 9)

    lammpstraj : str
            Path to the trajectory file.
    """

    def __init__(self, obj, header_lines=9, lammpstraj=None):
        """
        Python class constructor
        """

        super().__init__(obj, header_lines)  # fill the parent class
        self.lammpstraj = lammpstraj  # lammps file to read from.

    def process_trajectory_file(self, update_class=True):
        """ Get additional information from the trajectory file

        In this method, there are several doc string styled comments. This is included as there are several components
        of the method that are all related to the analysis of the trajectory file.

        Parameters
        ----------
        update_class : bool
                Boolean decision on whether or not to update the class. If yes, the full saved class instance will be
                updated with new information. This is necessary on the first run of data addition to the database. After
                this point, when new data is added, this is no longer required as other methods will take care of
                updating the properties that change with new data. In fact, it will set the number of configurations to
                only the new data, which will be wrong.
        """

        """
            Define necessary dicts and variables
        """
        species_summary = {}  # For storing the species or types of molecules
        n_lines_header_block = 9  # Standard header block of a lammpstraj file

        """
            Get the properties of each configuration
        """
        with open(self.project.trajectory_file) as f:

            """
                Get header files for analysis
            """
            head = [next(f).split() for _ in range(n_lines_header_block)]  # Get the first header
            f.seek(0)  # Go back to the start of the file
            number_of_atoms = int(head[3][0])  # Calculate the number of atoms

            """
                Fill data arrays with the first two configurations to get simulation properties
            """
            # Get first configuration
            first_configuration = [next(f).split() for _ in range(number_of_atoms + n_lines_header_block)]

            # Get the second configuration
            second_configuration = [next(f).split() for _ in range(number_of_atoms + n_lines_header_block)]

            """
                Calculate time properties of the simulation. Specifically, how often the configurations were dumped
                into the trajectory file. Very important for time calculations.
            """
            time_0 = float(first_configuration[1][0])  # Time in first configuration
            time_1 = float(second_configuration[1][0])  # Time in second configuration
            sample_rate = time_1 - time_0  # Chnage in time between the configurations

        """
            Calculate configuration and line properties of the simulation and determine the batch size
        """
        number_of_lines = line_counter(self.project.trajectory_file)  # get the number of lines in the file
        number_of_configurations = int(number_of_lines / (number_of_atoms + n_lines_header_block))  # n of timesteps
        batch_size = optimize_batch_size(self.project.trajectory_file, number_of_configurations)  # get the batch size

        """
            Get the position of the element keyword so that any format can be given. 
        """
        try:
            # Look for element keyword in trajectory.
            if "element" in first_configuration[8]:
                element_index = first_configuration[8].index("element") - 2

            # Look for type keyword if element is not present.
            elif "type" in first_configuration[8]:
                element_index = first_configuration[8].index('type') - 2

            # Raise an error if no identifying keywords are found.
            else:
                raise NoElementInDump
        except NoElementInDump:
            print("Insufficient species or type identification available.")

        """
            Get the species properties of the elements in the trajectory
        """
        # Loop over atoms in first configuration.
        for i in range(9, number_of_atoms + 9):

            # Check if the species is already in the summary.
            if first_configuration[i][element_index] not in species_summary:
                species_summary[first_configuration[i][element_index]] = {}
                species_summary[first_configuration[i][element_index]]['indices'] = []

            # Update the index of the atom in the summary.
            species_summary[first_configuration[i][element_index]]['indices'].append(i)

        """
            Get the available properties for analysis
        """
        header_line = first_configuration[8]  # the header line in the trajectory
        properties_summary = self._get_column_properties(header_line)  # get column properties

        """
            Get the box size from the first simulation cell
        """
        # TODO: Add this to the trajectory storing so that changing box sizes can be used
        box = [(float(first_configuration[5][1]) - float(first_configuration[5][0])),
               (float(first_configuration[6][1]) - float(first_configuration[6][0])),
               (float(first_configuration[7][1]) - float(first_configuration[7][0]))]

        """
            Update the class properties with those calculated above. 
        """
        if update_class:
            self.project.batch_size = batch_size
            self.project.dimensions = get_dimensionality(box)
            self.project.box_array = box
            self.project.volume = box[0] * box[1] * box[2]
            self.project.species = species_summary
            self.project.number_of_atoms = number_of_atoms
            self.project.properties = properties_summary
            self.project.number_of_configurations += number_of_configurations
            self.project.sample_rate = sample_rate

        else:
            self.project.batch_size = batch_size
            return [number_of_atoms, list(species_summary), box, number_of_configurations]

    @staticmethod
    def _get_column_properties(header_line):
        header_line = header_line[4:]
        properties_summary = {variable: idx + 2 for idx, variable in enumerate(header_line)}
        return properties_summary

    def _extract_properties(self):
        """
        Construct generalized property array

        Takes the lammps properties dictionary and constructs and array of properties which can be used by the species
        class.

        agrs:
            properties_dict (dict) -- A dictionary of all the available properties in the trajectory. This dictionary is
            built only from the LAMMPS symbols and therefore must be again processed to extract the useful information.

        returns:
            trajectory_properties (dict) -- A dictionary of the keyword labelled properties in the trajectory. The
            values of the dictionary keys correspond to the array location of the specific piece of data in the set.
        """

        # grab the properties present in the current case
        properties_dict = self.project.properties

        # for each property label (position, velocity,etc) in the lammps definition
        for property_label, property_names in lammps_properties_dict.items():
            # for each coordinate for a given property label (position: x, y, z), get idx and the name
            for idx, property_name in enumerate(property_names):
                if property_name in properties_dict.keys():  # if this name (x) is in the input file properties
                    # we change the lammps_properties_dict replacing the string of the property name by the column name
                    lammps_properties_dict[property_label][idx] = properties_dict[property_name]

        # trajectory_properties only needs the labels with the integer columns, then we one copy those
        trajectory_properties = {}
        for property_label, properties_columns in lammps_properties_dict.items():
            if all([isinstance(property_column, int) for property_column in properties_columns]):
                trajectory_properties[property_label] = properties_columns

        print("I have found the following properties with the columns in []: ")
        [print(key, value) for key, value in trajectory_properties.items()]

        return trajectory_properties

    def _read_lammpstrj(self):
        """ Process a lammps trajectory file """
        pass
