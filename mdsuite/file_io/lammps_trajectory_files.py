"""
Module for reading lammps trajectory files

Summary
-------
"""

from mdsuite.file_io.file_read import FileProcessor
# from mdsuite.file_io.file_io_dict import lammps_traj
from mdsuite.utils.exceptions import *
from mdsuite.utils.meta_functions import line_counter
from mdsuite.utils.meta_functions import optimize_batch_size
from mdsuite.utils.meta_functions import get_dimensionality
import numpy as np

lammps_traj = {
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
    "KE": ["c_KE"],
    "PE": ["c_PE"],
    "Stress": ['c_Stress[1]', 'c_Stress[2]', 'c_Stress[3]', 'c_Stress[4]', 'c_Stress[5]', 'c_Stress[6]']
}


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
        column_dict_properties = self._get_column_properties(header_line)  # get column properties
        # Get the observable groups
        self.project.property_groups = self._extract_properties(lammps_traj, column_dict_properties)

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
            self.project.number_of_configurations += number_of_configurations
            self.project.sample_rate = sample_rate

        else:
            self.project.batch_size = batch_size
            # return [number_of_atoms, list(species_summary), box, number_of_configurations]

    @staticmethod
    def _get_column_properties(header_line):
        header_line = header_line[2:]
        properties_summary = {variable: idx for idx, variable in enumerate(header_line)}
        return properties_summary

    def _read_lammpstrj(self):
        """ Process a lammps trajectory file """
        pass

    def process_configurations(self, data, database, counter):
        """
        Process the available data

        Called during the main database creation. This function will calculate the number of configurations
        within the raw data and process it.

        Parameters
        ----------
        data : np.array
                Array of the raw data for N configurations.

        database : object
                Database in which to store the data.

        counter : int
                Which configuration to start from.
        """

        # Re-calculate the number of available configurations for analysis
        partitioned_configurations = int(len(data) / self.project.number_of_atoms)

        for item in self.project.species:
            """
            Get the new indices for the positions. This function requires the atoms to be in the same position during
            each configuration. The calculation simply adds multiples of the number of atoms and configurations to the
            position of each atom in order to read the correct part of the file.
            """
            # TODO: Implement a sort algorithm or something of the same kind.
            positions = np.array([np.array(self.project.species[item]['indices']) + i * self.project.number_of_atoms -
                                  self.header_lines for i in range(int(partitioned_configurations))]).flatten()

            """
            Fill the database
            """
            axis_names = ('x', 'y', 'z', 'xy', 'xz', 'yz')
            # Fill the database
            for property_group, columns in self.project.property_groups.items():
                num_columns = len(columns)
                if num_columns == 1:
                    database[item][property_group][:, counter:counter + partitioned_configurations] = \
                        data[positions][:, columns[0]].astype(float).reshape(
                            (len(self.project.species[item]['indices']), partitioned_configurations), order='F')
                else:
                    for column, axis in zip(columns, axis_names):
                        database[item][property_group][axis][:, counter:counter + partitioned_configurations] = \
                            data[positions][:, column].astype(float).reshape(
                                (len(self.project.species[item]['indices']), partitioned_configurations), order='F')