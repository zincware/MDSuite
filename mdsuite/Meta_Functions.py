"""
Author: Samuel Tovey
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Purpose: This file contains arbitrary functions used in several different processes. They are often generic and serve
         smaller purposes in order to clean up code in more important parts of the program.
"""

import os
import psutil

def Write_XYZ(data):
    """ Write an xyz file from data arrays

    For some of the properties calculated it is beneficial to have an xyz file for analysis with other platforms.
    This function will write an xyz file from a numpy array of some property. Can be used in the visualization of
    trajectories.

    args:
        data (array) -- Array of property to be studied
    """

    number_of_configurations: int = int(len(data[0][0])) # Get number of configurations
    species = ["Na", "Cl"]

    # Calculate the number of atoms in the full system
    number_of_atoms: int = 0
    for i in range(len(data)):
        number_of_atoms += len(data[i])

    write_array = []

    # Construct the write array
    for i in range(number_of_configurations):
        write_array.append(number_of_atoms)
        write_array.append("Nothing to see here")
        for j in range(len(data)):
            for k in range(len(data[j])):
                write_array.append("{0:<}    {1:>9.4f}    {2:>9.4f}    {3:>9.4f}".format(species[j],
                                                                                    data[j][k][i][0],
                                                                                    data[j][k][i][1],
                                                                                    data[j][k][i][2]))

    # Write the array to an output file
    with open("output.xyz", "w") as f:
        for line in write_array:
            f.write("%s\n" % line)

def Get_Dimensionality(box):
    """ Calculate the dimensionality of the system box

    args:
        box (list) -- box array [x, y, z]

    returns:
        dimensions (int) -- dimension of the box i.e, 1 or 2 or 3 (Higher dimensions probably don't make sense just yet)
    """

    if box[0] == 0 or box[1] == 0 or box[2] == 0:
        dimensions = 2

    elif box[0] == 0 and box[1] == 0 or box[0] == 0 and box[2] == 0 or box[1] == 0 and box[2] == 0:
        dimensions = 1

    else:
        dimensions = 3

    return dimensions

def Extract_LAMMPS_Properties(properties_dict):
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
    LAMMPS_Properties = ["Positions", "Scaled_Positions", "Unwrapped_Positions", "Scaled_Unwrapped_Positions",
                         "Velocities", "Forces", "Box_Images", "Dipole_Orientation_Magnitude", "Angular_Velocity_Spherical",
                         "Angular_Velocity_Non_Spherical", "Torque"]
    trajectory_properties = {}
    system_properties = list(properties_dict)

    if 'x' in system_properties:
        trajectory_properties[LAMMPS_Properties[0]] = [properties_dict['x'],
                                                   properties_dict['y'],
                                                   properties_dict['z']]
    if 'xs' in system_properties:
        trajectory_properties[LAMMPS_Properties[1]] = [properties_dict['xs'],
                                                   properties_dict['ys'],
                                                   properties_dict['zs']]
    if 'xu' in system_properties:
        trajectory_properties[LAMMPS_Properties[2]] = [properties_dict['xu'],
                                                   properties_dict['yu'],
                                                   properties_dict['zu']]
    if 'xsu' in system_properties:
        trajectory_properties[LAMMPS_Properties[3]] = [properties_dict['xsu'],
                                                   properties_dict['ysu'],
                                                   properties_dict['zsu']]
    if 'vx' in system_properties:
        trajectory_properties[LAMMPS_Properties[4]] = [properties_dict['vx'],
                                                   properties_dict['vy'],
                                                   properties_dict['vz']]
    if 'fx' in system_properties:
        trajectory_properties[LAMMPS_Properties[5]] = [properties_dict['fx'],
                                                   properties_dict['fy'],
                                                   properties_dict['fz']]
    if 'ix' in system_properties:
        trajectory_properties[LAMMPS_Properties[6]] = [properties_dict['ix'],
                                                   properties_dict['iy'],
                                                   properties_dict['iz']]
    if 'mux' in system_properties:
        trajectory_properties[LAMMPS_Properties[7]] = [properties_dict['mux'],
                                                   properties_dict['muy'],
                                                   properties_dict['muz']]
    if 'omegax' in system_properties:
        trajectory_properties[LAMMPS_Properties[8]] = [properties_dict['omegax'],
                                                   properties_dict['omegay'],
                                                   properties_dict['omegaz']]
    if 'angmomx' in system_properties:
        trajectory_properties[LAMMPS_Properties[9]] = [properties_dict['angmomx'],
                                                   properties_dict['angmomy'],
                                                   properties_dict['angmomz']]
    if 'tqx' in system_properties:
        trajectory_properties[LAMMPS_Properties[10]] = [properties_dict['tqx'],
                                                    properties_dict['tqy'],
                                                    properties_dict['tqz']]

    return trajectory_properties

def Extract_extxyz_Properties(properties_dict):
    """ Construct generalized property array

    Takes the extxyz properties dictionary and constructs and array of properties which can be used by the species
    class.
    """

    # Define Initial Properties and arrays
    extxyz_properties = ['Positions', 'Forces']
    output_properties = []
    system_properties = list(properties_dict)

    if 'pos' in system_properties:
        output_properties.append(extxyz_properties[0])
    if 'force' in system_properties:
        output_properties.append(extxyz_properties[1])

    return output_properties

def Line_Counter(filename):
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)

    return lines

def Optimize_Batch_Size(filepath, number_of_configurations):
    """ Optimize the size of batches during initial processing

    During the database construction a batch size must be chosen in order to process the trajectories with the
    least RAM but reasonable performance.
    """

    file_size = os.path.getsize(filepath) # Get the size of the file
    available_memory = psutil.virtual_memory().available
    memory_per_configuration = file_size / number_of_configurations # get the memory per configuration
    database_memory = 0.1*available_memory # We take 20% of the available memory
    initial_batch_number = int(database_memory / memory_per_configuration) # trivial batch allocation

    if file_size < database_memory:
        batch_number = number_of_configurations
    else:
        remainder = 1000000000
        for i in range(10):
            r_temp = number_of_configurations % (initial_batch_number - i)
            if r_temp <= remainder:
                batch_number = initial_batch_number - i

        if batch_number > 1000:
            batch_number = 1000

    return batch_number

def Linear_Fitting_Function(x, a, b):
    """ Linear function for line fitting

    In many cases, namely those involving an Einstein relation, a linear curve must be fit to some data. This function
    is called by the scipy curve_fit module as the model to fit to.

    args:
        x (list) -- x data for fitting
        a (float) -- fitting parameter of the gradient
        b (float) -- fitting parameter for the y intercept
    """
    return a*x + b

















