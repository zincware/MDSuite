"""
Author: Samuel Tovey ; Francisco Torres
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Purpose: Larger methods used in the Experiment class
"""
import pickle

import h5py as hf
import pubchempy as pcp
import json
import numpy as np

import mdsuite.utils.constants as constants

# Using static file from data
from importlib.resources import open_text
from mdsuite import data as static_data

class ProjectMethods:
    """ methods to be used in the Experiment class """

    def build_species_dictionary(self):
        """ Add information to the species dictionary

        A fundamental part of this package is species specific analysis. Therefore, the mendeleev python package is
        used to add important species specific information to the class. This will include the charge of the ions which
        will be used in conductivity calculations.

        returns:
            This method will update the class attributes in place and therefore, will not return anything explicitly.
        """
        with open_text(static_data, 'PubChemElements_all.json') as json_file:
            PSE = json.loads(json_file.read())

            # Try to get the species data from the Periodic System of Elements file
        for element in self.species:
            self.species[element]['charge'] = [0.0]
            for entry in PSE:
                if PSE[entry][1] == element:
                    self.species[element]['mass'] = [float(PSE[entry][3])]

            # If gathering the data from the PSE file was not succesfull try to get it from Pubchem via pubchempy
        for element in self.species:
            if not 'mass' in self.species[element]:
                try:
                    temp = pcp.get_compounds(element, 'name')
                    temp[0].to_dict(properties=['atoms', 'bonds', 'exact_mass', 'molecular_weight', 'elements'])
                    self.species[element]['mass'] = temp[0].molecular_weight
                    print(temp[0].exact_mass)
                except:
                    self.species[element]['mass'] = [0.0]
                    print(f'WARNING element {element} has been assigned mass=0.0')

    def _build_database_skeleton(self):
        """ Build skeleton of the hdf5 database

        Gathers all of the properties of the system using the relevant functions. Following the gathering
        of the system properties, this function will read through the first configuration of the dataset, and
        generate the necessary database structure to allow for the following generation to take place. This will
        include the separation of species, atoms, and properties. For a full description of the data structure,
        look into the documentation.
        """

        database = hf.File('{0}/{1}/{1}.hdf5'.format(self.filepath, self.analysis_name), 'w', libver='latest')

        property_groups = meta_functions.extract_lammps_properties(self.properties)  # Get the property groups
        self.property_groups = property_groups

        # Build the database structure
        for item in self.species:
            database.create_group(item)
            for property in property_groups:
                database[item].create_group(property)
                database[item][property].create_dataset("x", (
                len(self.species[item]['indices']), self.number_of_configurations -
                self.number_of_configurations % self.batch_size),
                                                        compression="gzip", compression_opts=9)
                database[item][property].create_dataset("y", (
                len(self.species[item]['indices']), self.number_of_configurations -
                self.number_of_configurations % self.batch_size),
                                                        compression="gzip", compression_opts=9)
                database[item][property].create_dataset("z", (
                len(self.species[item]['indices']), self.number_of_configurations -
                self.number_of_configurations % self.batch_size),
                                                        compression="gzip", compression_opts=9)

    def process_configurations(self, data, database, counter):
        """ Process the available data

        Called during the main database creation. This function will calculate the number of configurations within the
        raw data and process it.

        args:
            data (numpy array) -- Array of the raw data for N configurations.
            database (object) --
            counter (int) --
        """

        # Re-calculate the number of available configurations for analysis
        partitioned_configurations = int(len(data) / self.number_of_atoms)

        for item in self.species:
            # get the new indices for the positions
            positions = np.array([np.array(self.species[item]['indices']) + i * self.number_of_atoms - 9 for i in
                                  range(int(partitioned_configurations))]).flatten()
            # Fill the database
            for property_group in self.property_groups:
                database[item][property_group]["x"][:, counter:counter + partitioned_configurations] = \
                    data[positions][:, self.property_groups[property_group][0]].astype(float).reshape(
                        (len(self.species[item]['indices']), partitioned_configurations), order='F')

                database[item][property_group]["y"][:, counter:counter + partitioned_configurations] = \
                    data[positions][:, self.property_groups[property_group][1]].astype(float).reshape(
                        (len(self.species[item]['indices']), partitioned_configurations), order='F')

                database[item][property_group]["z"][:, counter:counter + partitioned_configurations] = \
                    data[positions][:, self.property_groups[property_group][2]].astype(float).reshape(
                        (len(self.species[item]['indices']), partitioned_configurations), order='F')

    def print_data_structrure(self):
        """ Print the data structure of the hdf5 dataset """

        database = hf.File("{0}/{1}/{1}.hdf5".format(self.filepath, self.analysis_name), "r")

    def write_xyz(self, property="Positions", species=None):
        """ Write an xyz file from database array

        For some of the properties calculated it is beneficial to have an xyz file for analysis with other platforms.
        This function will write an xyz file from a numpy array of some property. Can be used in the visualization of
        trajectories.

        kwargs:
            property (str) -- Which property would you like to print
            species (list) -- List of species for which you would like to write the file
        """

        if species is None:
            species = list(self.species)

        data_matrix = self.load_matrix(property, species)

        with open(f"{self.filepath}/{self.analysis_name}/{property}_{'_'.join(species)}.xyz", 'w') as f:
            for i in range(self.number_of_configurations):
                f.write(f"{self.number_of_atoms}\n")
                f.write("Generated by the mdsuite xyz writer\n")
                for j in range(len(species)):
                    for atom in data_matrix[j]:
                        f.write(f"{species[j]:<2}    {atom[i][0]:>9.4f}    {atom[i][1]:>9.4f}    {atom[i][2]:>9.4f}\n")

    def _save_class(self):
        """ Saves class instance

        In order to keep properties of a class the state must be stored. This method will store the instance of the
        class for later re-loading
        """

        save_file = open(f"{self.storage_path}/{self.analysis_name}/{self.analysis_name}.bin", 'wb')
        save_file.write(pickle.dumps(self.__dict__))
        save_file.close()

    def _load_class(self):
        """ Load class instance

        A function to load a class instance given the project name.
        """

        class_file = open(f'{self.storage_path}/{self.analysis_name}/{self.analysis_name}.bin', 'rb')
        pickle_data = class_file.read()
        class_file.close()

        self.__dict__ = pickle.loads(pickle_data)

    def print_class_attributes(self):
        """ Print all attributes of the class """

        attributes = []
        for item in vars(self).items():
            attributes.append(item)
        for tuple in attributes:
            print(f"{tuple[0]}: {tuple[1]}")

        return attributes

    @staticmethod
    def units_to_si(units_system):
        """ Passes the given dimension to SI units.

        It is easier to work in SI units always, to avoid mistakes.

        Parameters
        ----------
        units_system (str) -- current unit system
        dimension (str) -- dimension you would like to change

        Returns
        -------
        conv_factor (float) -- conversion factor to pass to SI

        Examples
        --------
        Pass from metal units of time (ps) to SI

        >>> units_to_si('metal')
        {'time': 1e-12, 'length': 1e-10, 'energy': 1.6022e-19}
        """
        units = {
            "metal": {'time': 1e-12, 'length': 1e-10, 'energy': 1.6022e-19},
            "real": {'time': 1e-15, 'length': 1e-10, 'energy': 4184 / constants.avogadro_constant},
        }

        return units[units_system]
