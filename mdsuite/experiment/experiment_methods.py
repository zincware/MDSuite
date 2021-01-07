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
from importlib.resources import open_text

from diagrams import Diagram, Cluster
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.network import ELB
from diagrams.aws.compute import ECS

from diagrams.generic.place import Datacenter
from diagrams.generic.storage import Storage
from diagrams.generic.database import SQL

import mdsuite.utils.constants as constants
from mdsuite import data as static_data

class ProjectMethods:
    """
    Methods to be used in the Experiment class
    """

    def build_species_dictionary(self):
        """
        Add information to the species dictionary

        Summary
        -------
        A fundamental part of this package is species specific analysis. Therefore, the mendeleev python package is
        used to add important species specific information to the class. This will include the charge of the ions which
        will be used in conductivity calculations.

        """
        with open_text(static_data, 'PubChemElements_all.json') as json_file:
            PSE = json.loads(json_file.read())

        # Try to get the species data from the Periodic System of Elements file
        for element in self.species:
            self.species[element]['charge'] = [0.0]
            for entry in PSE:
                if PSE[entry][1] == element:
                    self.species[element]['mass'] = [float(PSE[entry][3])]

        # If gathering the data from the PSE file was not successful try to get it from Pubchem via pubchempy
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

    def set_element(self, old_name, new_name):
        """
        Change the name of the element in the self.species dictionary

        Parameters
        ----------
        old_name : str
                Name of the element you want to change
        new_name : str
                New name of the element
        """
        # Check if the new name is new
        if new_name != old_name:
            self.species[new_name] = self.species[old_name]  # update dict
            del self.species[old_name]  # remove old entry

    def set_charge(self, element, charge):
        """
        Set the charge/s of an element

        Parameters
        ----------
        element : str
                Name of the element whose charge you want to change
        charge : list
                New charge/s of the element
        """
        self.species[element]['charge'] = charge  # update entry

    def set_mass(self, element, mass):
        """
        Set the mass/es of an element

        Parameters
        ----------
        element : str
                Name of the element whose mass you want to change
        mass : list
                New mass/es of the element
        """
        self.species[element]['mass'] = mass  # update the mass

    def print_data_structure(self):
        """
        Print the data structure of the hdf5 dataset
        """

        database = hf.File("{0}/{1}/{1}.hdf5".format(self.filepath, self.analysis_name), "r")
        with Diagram("Web Service", show=True, direction='TB'):
            head = RDS("Database")  # set the head database object
            for item in database:
                with Cluster(f"{item}"):
                    group_list = []
                    for property_group in database[item]:
                        group_list.append(ECS(property_group))  # construct a list of equal level groups
                head >> group_list  # append these groups to the head object

    def write_xyz(self, property="Positions", species=None):
        """
        Write an xyz file from database array

        Summary
        -------
        For some of the properties calculated it is beneficial to have an xyz file for analysis with other platforms.
        This function will write an xyz file from a numpy array of some property. Can be used in the visualization of
        trajectories.

        Parameters
        ----------
        property : str
                Which property would you like to print
        species : list
                List of species for which you would like to write the file

        Notes
        -----
        Currently this is not memory safe. Only use it if the full trajectory can be fit into memory.
        """

        # If no species are given, use all available species.
        if species is None:
            species = list(self.species)

        # TODO: Make memory safe - load in batches
        data_matrix = self.load_matrix(property, species)  # load the matrix of data

        with open(f"{self.filepath}/{self.analysis_name}/{property}_{'_'.join(species)}.xyz", 'w') as f:
            for i in range(self.number_of_configurations):
                f.write(f"{self.number_of_atoms}\n")  # add the first header line
                f.write("Generated by the mdsuite xyz writer\n")  # add comment line

                # Add the atoms data
                for j in range(len(species)):
                    for atom in data_matrix[j]:
                        f.write(f"{species[j]:<2}    {atom[i][0]:>9.4f}    {atom[i][1]:>9.4f}    {atom[i][2]:>9.4f}\n")

    def _save_class(self):
        """
        Saves class instance

        Summary
        -------
        In order to keep properties of a class the state must be stored. This method will store the instance of the
        class for later re-loading
        """

        save_file = open(f"{self.storage_path}/{self.analysis_name}/{self.analysis_name}.bin", 'wb')  # construct file
        save_file.write(pickle.dumps(self.__dict__))                                                  # write to file
        save_file.close()                                                                             # close file.

    def load_class(self):
        """
        Load class instance

        Summary
        -------
        A function to load a class instance given the project name.
        """

        class_file = open(f'{self.storage_path}/{self.analysis_name}/{self.analysis_name}.bin', 'rb')  # open file
        pickle_data = class_file.read()                                                                # read file
        class_file.close()                                                                             # close file

        self.__dict__ = pickle.loads(pickle_data)  # update the class object

    def print_class_attributes(self):
        """
        Print all attributes of the class

        Returns
        -------
        attributes : list
                List of class attribute tuples of (key, value)
        """

        attributes = []                       # define empty array
        for item in vars(self).items():       # loop over class attributes
            attributes.append(item)           # append to the attributes array
        for tuple in attributes:              # Split the key and value terms
            print(f"{tuple[0]}: {tuple[1]}")  # Format the print statement

        return attributes

    @staticmethod
    def units_to_si(units_system):
        """
        Passes the given dimension to SI units.

        Summary
        -------
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
