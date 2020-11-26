""" Module for reading lammps trajectory files"""

from mdsuite.file_io.file_read import FileProcessor
from mdsuite.utils.constants import lammps_properties_labels
from mdsuite.utils.constants import lammps_properties
from mdsuite.utils.exceptions import *
from mdsuite.utils.meta_functions import line_counter
from mdsuite.utils.meta_functions import optimize_batch_size
from mdsuite.utils.meta_functions import simple_file_read
from mdsuite.utils.meta_functions import get_dimensionality


class LAMMPSTrajectoryFile(FileProcessor):
    """ Child class for the lammps file reader """

    def __init__(self, obj, header_lines=9, lammpstraj=None):
        """ Python class constructor """

        super().__init__(obj, header_lines)
        self.lammpstraj = lammpstraj

    def process_trajectory_file(self, update_class=True):
        """ Get additional information from the trajectory file

        In this method, there are several doc string styled comments. This is included as there are several components
        of the method that are all related to the analysis of the trajectory file.
        """

        """
            Define necessary dicts and variables
        """
        species_summary = {}  # For storing the species or types of molecules
        properties_summary = {}  # For the storing of properties
        n_lines_header_block = 9  # Standard header block of a lammps traj file

        """
            Get the properties of each configuration
        """
        with open(self.project.trajectory_file) as f:

            """
                Get header files for analysis
            """
            head = [next(f).split() for _ in range(n_lines_header_block)]
            f.seek(0)  # Go back to the start of the file
            # Calculate the number of atoms and configurations in the system
            number_of_atoms = int(head[3][0])

            """
                Fill data arrays with the first two configurations to get simulation properties
            """
            # Get first configuration
            first_configuration = [next(f).split() for _ in range(number_of_atoms + n_lines_header_block)]

            # Get the second configuration
            second_configuration = [next(f).split() for _ in range(number_of_atoms + n_lines_header_block)]

            """
                Calculate time properties of the simulation
            """
            time_0 = float(first_configuration[1][0])
            time_1 = float(second_configuration[1][0])
            sample_rate = time_1 - time_0

        """
            Calculate configuration and line properties of the simulation and determine the batch size
        """
        number_of_lines = line_counter(self.project.trajectory_file)
        number_of_configurations = int(number_of_lines / (number_of_atoms + n_lines_header_block))  # n of timesteps
        batch_size = optimize_batch_size(self.project.trajectory_file, number_of_configurations)

        """
            Get the position of the element keyword so that any format can be given. 
        """
        try:
            if "element" in first_configuration[8]:
                element_index = first_configuration[8].index("element") - 2
            elif "type" in first_configuration[8]:
                element_index = first_configuration[8].index('type') - 2
            else:
                raise NoElementInDump
        except:
            print("Insufficient species or type identification available.")

        """
            Get the species properties of the elements in the trajectory
        """
        for i in range(9, number_of_atoms + 9):
            if first_configuration[i][element_index] not in species_summary:
                species_summary[first_configuration[i][element_index]] = {}
                species_summary[first_configuration[i][element_index]]['indices'] = []

            species_summary[first_configuration[i][element_index]]['indices'].append(i)

        """
            Get the available properties for analysis
        """
        for i in range(len(first_configuration[8])):
            if first_configuration[8][i] in lammps_properties_labels:
                properties_summary[first_configuration[8][i]] = i - 2

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

    def _extract_properties(self):
        """ Construct generalized property array

            Takes the lammps properties dictionary and constructs and array of properties which can be used by the species
            class.

            agrs:
                properties_dict (dict) -- A dictionary of all the available properties in the trajectory. This dictionary is
                built only from the LAMMPS symbols and therefore must be again processed to extract the useful information.

            returns:
                trajectory_properties (dict) -- A dictionary of the keyword labelled properties in the trajectory. The
                values of the dictionary keys correspond to the array location of the specific piece of data in the set.
            """

        # Define Initial Properties and arrays

        trajectory_properties = {}
        system_properties = list(self.project.properties)
        properties_dict = self.project.properties

        if 'x' in system_properties:
            trajectory_properties[lammps_properties[0]] = [properties_dict['x'],
                                                           properties_dict['y'],
                                                           properties_dict['z']]
        if 'xs' in system_properties:
            trajectory_properties[lammps_properties[1]] = [properties_dict['xs'],
                                                           properties_dict['ys'],
                                                           properties_dict['zs']]
        if 'xu' in system_properties:
            trajectory_properties[lammps_properties[2]] = [properties_dict['xu'],
                                                           properties_dict['yu'],
                                                           properties_dict['zu']]
        if 'xsu' in system_properties:
            trajectory_properties[lammps_properties[3]] = [properties_dict['xsu'],
                                                           properties_dict['ysu'],
                                                           properties_dict['zsu']]
        if 'vx' in system_properties:
            trajectory_properties[lammps_properties[4]] = [properties_dict['vx'],
                                                           properties_dict['vy'],
                                                           properties_dict['vz']]
        if 'fx' in system_properties:
            trajectory_properties[lammps_properties[5]] = [properties_dict['fx'],
                                                           properties_dict['fy'],
                                                           properties_dict['fz']]
        if 'ix' in system_properties:
            trajectory_properties[lammps_properties[6]] = [properties_dict['ix'],
                                                           properties_dict['iy'],
                                                           properties_dict['iz']]
        if 'mux' in system_properties:
            trajectory_properties[lammps_properties[7]] = [properties_dict['mux'],
                                                           properties_dict['muy'],
                                                           properties_dict['muz']]
        if 'omegax' in system_properties:
            trajectory_properties[lammps_properties[8]] = [properties_dict['omegax'],
                                                           properties_dict['omegay'],
                                                           properties_dict['omegaz']]
        if 'angmomx' in system_properties:
            trajectory_properties[lammps_properties[9]] = [properties_dict['angmomx'],
                                                           properties_dict['angmomy'],
                                                           properties_dict['angmomz']]
        if 'tqx' in system_properties:
            trajectory_properties[lammps_properties[10]] = [properties_dict['tqx'],
                                                            properties_dict['tqy'],
                                                            properties_dict['tqz']]

        return trajectory_properties

    def _read_lammpstrj(self):
        """ Process a lammps trajectory file """
