import numpy as np
import os
import h5py as hf
import psutil as ps
import Meta_Functions

#filename = "test_simulatin.xyz"
filename = "/beegfs/work/stovey/LAMMPSSims/NaCl/scaledSim/10000Atoms/NaCl_Velocities.xyz"
#filename = "/beegfs/work/stovey/LAMMPSSims/NaCl/scaledSim/Thesis_Sims/1400K/NaCl_Velocities.xyz"

#database = hf.File('test_database.hdf5', 'w')

p = ps.Process()


def Get_LAMMPS_Properties(data_array):
    """ Get the properties of the system from a custom lammps dump file

        args:
            data_array (list) -- Array containing trajectory data

        returns:
            species_summary (dict) -- Dictionary containing all the species in the systems
                                      and how many of them there are in each configuration.
            properties_summary (dict) -- All the properties available in the dump file for
                                         analysis and their index in the file
    """

    # Define necessary properties and attributes
    species_summary = {}
    properties_summary = {}
    LAMMPS_Properties_labels = {'x', 'y', 'z',
                                'xs', 'ys', 'zs',
                                'xu', 'yu', 'zu',
                                'xsu', 'ysu', 'zsu',
                                'ix', 'iy', 'iz',
                                'vx', 'vy', 'vz',
                                'fx', 'fy', 'fz',
                                'mux', 'muy', 'muz', 'mu',
                                'omegax', 'omegay', 'omegaz',
                                'angmomx', 'angmomy', 'angmomz',
                                'tqx', 'tqy', 'tqz'}

    # Calculate the number of atoms and configurations in the system
    number_of_atoms = int(data_array[3][0])
    number_of_configurations = int(len(data_array) / (number_of_atoms + 9))

    # Find the information regarding species in the system and construct a dictionary
    for i in range(9, number_of_atoms + 9):
        if data_array[i][2] not in species_summary:
            species_summary[data_array[i][2]] = []

        species_summary[data_array[i][2]].append(i)

    # Find properties available for analysis
    for i in range(len(data_array[8])):
        if data_array[8][i] in LAMMPS_Properties_labels:
            properties_summary[data_array[8][i]] = i - 2

    # Get the box size from the system
    box = [(float(data_array[5][1][:-10]) - float(data_array[5][0][:-10])) * 10,
           (float(data_array[6][1][:-10]) - float(data_array[6][0][:-10])) * 10,
           (float(data_array[7][1][:-10]) - float(data_array[7][0][:-10])) * 10]

    # Update class attributes with calculated data
    return Meta_Functions.Get_Dimensionality(box), box, box[0] * box[1] * box[2], species_summary, number_of_atoms, properties_summary, number_of_configurations


def Build_Database_Skeleton():
    """ Build skeleton of the hdf5 database

    Gathers all of the properties of the system using the relevant functions. Following the gathering
    of the system properties, this function will read through the first configuration of the dataset, and
    generate the necessary database structure to allow for the following generation to take place. This will
    include the separation of species, atoms, and properties. For a full description of the data structure,
    look into the documentation.
    """

    database = hf.File('test_database.hdf5', 'w')

    with open(filename) as f:
        head = [next(f).split() for i in range(9)] # Get the meta-data
        f.seek(0) # Go back to the start of the file

        first_configuration = [next(f).split() for i in range(int(head[3][0]) + 9)] # Get first configuration
        f.seek(0) # Back to start of the file

        dimension, box, volume, species_summary, number_of_atoms, properties_summary, number_of_configurations = Get_LAMMPS_Properties(first_configuration) # Process this configuration
        property_groups = Meta_Functions.Extract_LAMMPS_Properties(properties_summary) # Get the property groups

        # Build the database structure
        for item in list(species_summary.keys()):
            database.create_group(item)
            for property in property_groups:
                database[item].create_group(property)
                for index in species_summary[item]:
                    database[item][property].create_dataset(str(index - 8), (dimension, number_of_configurations),
                                                            compression="gzip")



Build_Database_Skeleton()


