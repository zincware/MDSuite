"""
Module for reading lammps trajectory files

Summary
-------
"""

from mdsuite.file_io.trajectory_files import TrajectoryFile
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


class LAMMPSTrajectoryFile(TrajectoryFile):
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
        """
        Get additional information from the trajectory file

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
        column_dict_properties = self._get_column_properties(header_line, skip_words=2)  # get column properties, skip the two first words which are not in the columns (ITEM: ATOMS)
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

