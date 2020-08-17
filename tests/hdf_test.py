import numpy as np
import os
import h5py as hf
import psutil as ps
import Meta_Functions
import time
from itertools import islice
from multiprocessing import Process, Value, Lock
import multiprocessing as mp
import pandas as pd

#filename = "test_simulatin.xyz"
#filename = "/beegfs/work/stovey/LAMMPSSims/NaCl/scaledSim/10000Atoms/NaCl_Velocities.xyz"
filename = "/beegfs/work/stovey/LAMMPSSims/NaCl/scaledSim/Thesis_Sims/1400K/rerun/NaCl_Velocities.xyz"
number_of_atoms = 1000
number_of_configurations = 500000 #228 #229 #240001 # 229
labels = ['id', 'type', 'element', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz']
species_summary = {'Na': [i for i in range(500)], 'Cl': [i for i in range(500, 1000)]}




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

        species_summary[data_array[i][2]].append(i-9)

    # Find properties available for analysis
    labels = data_array[8][2:] # get the column labels in the data
    for i in range(len(labels)):
        if labels[i] in LAMMPS_Properties_labels:
            properties_summary[labels[i]] = i - 2

    # Get the box size from the system
    box = [(float(data_array[5][1][:-10]) - float(data_array[5][0][:-10])) * 10,
           (float(data_array[6][1][:-10]) - float(data_array[6][0][:-10])) * 10,
           (float(data_array[7][1][:-10]) - float(data_array[7][0][:-10])) * 10]

    # Update class attributes with calculated data
    return Meta_Functions.Get_Dimensionality(box), box, box[0] * box[1] * box[2], species_summary, number_of_atoms, properties_summary, number_of_configurations, labels


def Build_Database_Skeleton():
    """ Build skeleton of the hdf5 database

    Gathers all of the properties of the system using the relevant functions. Following the gathering
    of the system properties, this function will read through the first configuration of the dataset, and
    generate the necessary database structure to allow for the following generation to take place. This will
    include the separation of species, atoms, and properties. For a full description of the data structure,
    look into the documentation.
    """

    database = hf.File('test_database.hdf5', 'w', libver='latest')

    with open(filename) as f:
        head = [next(f).split() for i in range(9)] # Get the meta-data
        f.seek(0) # Go back to the start of the file

        first_configuration = [next(f).split() for i in range(int(head[3][0]) + 9)] # Get first configuration
        f.seek(0) # Back to start of the file

        (dimension, box, volume, species_summary, number_of_atoms,
         properties_summary, number_of_configurations, labels) = Get_LAMMPS_Properties(first_configuration)


        property_groups = Meta_Functions.Extract_LAMMPS_Properties(properties_summary) # Get the property groups

        number_of_configurations = 500000


        #Build the database structure
        for item in list(species_summary.keys()):
            database.create_group(item)
            for property in property_groups:
                database[item].create_group(property)
                database[item][property].create_dataset("x", (len(species_summary[item]), number_of_configurations),
                                                        compression = "gzip", compression_opts=9)
                database[item][property].create_dataset("y", (len(species_summary[item]), number_of_configurations),
                                                        compression = "gzip", compression_opts=9)
                database[item][property].create_dataset("z", (len(species_summary[item]), number_of_configurations),
                                                        compression = "gzip", compression_opts=9)


def Read_Configurations(N, f):
    """ Read in N configurations

    This function will read in N configurations from the file that has been opened previously by the parent method.

    N (int) -- Number of configurations to read in. This will depend on memory availability and the size of each
                configuration. Automatic setting of this variable is not yet available and therefore, it will be set
                manually.
    """
    data = []

    for i in range(N):

        # Skip header lines
        for j in range(9):
            f.readline()

        for k in range(number_of_atoms):
            data.append(f.readline().split())

    return np.array(data)


def Process_Configurations(data, database, counter):
    """ Process the available data

    Called during the main database creation. This function will calculate the number of configurations within the raw
    data and process it.

    args:
        data (numpy array) -- Array of the raw data for N configurations.
    """

    # Re-calculate the number of available configurations for analysis
    partitioned_configurations = int(len(data)/number_of_atoms)
    
    start = time.time()
    for item in species_summary:
        positions = np.array([np.array(species_summary[item]) + i*number_of_atoms for i in range(int(partitioned_configurations))]).flatten()
        #for i in range(int(partitioned_configurations)):
        #positions = np.array(species_summary[item]) + i*number_of_atoms

        database[item]["Positions"]["x"][:, counter:counter + partitioned_configurations] = data[positions][:, 3].astype(float).reshape((500, partitioned_configurations), order='F')
        database[item]["Positions"]["y"][:, counter:counter + partitioned_configurations] = data[positions][:, 4].astype(float).reshape((500, partitioned_configurations), order='F')
        database[item]["Positions"]["z"][:, counter:counter + partitioned_configurations] = data[positions][:, 5].astype(float).reshape((500, partitioned_configurations), order='F')

        database[item]["Velocities"]["x"][:, counter:counter + partitioned_configurations] = data[positions][:, 6].astype(float).reshape((500, partitioned_configurations), order='F')
        database[item]["Velocities"]["y"][:, counter:counter + partitioned_configurations] = data[positions][:, 7].astype(float).reshape((500, partitioned_configurations), order='F')
        database[item]["Velocities"]["z"][:, counter:counter + partitioned_configurations] = data[positions][:, 8].astype(float).reshape((500, partitioned_configurations), order='F')

        database[item]["Forces"]["x"][:, counter:counter + partitioned_configurations] = data[positions][:, 9].astype(float).reshape((500, partitioned_configurations), order='F')
        database[item]["Forces"]["y"][:, counter:counter + partitioned_configurations] = data[positions][:, 10].astype(float).reshape((500, partitioned_configurations), order='F')
        database[item]["Forces"]["z"][:, counter:counter + partitioned_configurations] = data[positions][:, 11].astype(float).reshape((500, partitioned_configurations), order='F')

def Build_LAMMPS_Database():
    """ Construct LAMMPS database from skeleton """

    database = hf.File("test_database.hdf5", "r+")

    with open(filename) as f:

        counter = 0
        for i in range(int(number_of_configurations/1000)):
            print("{0}% Complete".format(counter/number_of_configurations))
            test = Read_Configurations(1000, f)

            Process_Configurations(test, database, counter)

            counter += 1000


def Open_and_Read():

    database = hf.File("test_database.hdf5", 'r')

    print(database["Na"]["Positions"]["x"])

if __name__ == "__main__":
    start = time.time()
    Build_Database_Skeleton()
    Build_LAMMPS_Database()
    Open_and_Read()
    end = time.time()
    print("Elapsed = %s" % (end - start))


