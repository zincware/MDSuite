"""
Author: Samuel Tovey
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Purpose: This file contains arbitrary functions used in several different processes. They are often generic and serve
         smaller purposes in order to clean up code in more important parts of the program.
"""

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
    """

    # Define Initial Properties and arrays
    LAMMPS_Properties = ["Positions", "Scaled_Positions", "Unwrapped_Positions", "Scaled_Unwrapped_Positions",
                         "Velocities", "Forces", "Box_Images", "Dipole_Orientation_Magnitude", "Angular_Velocity_Spherical",
                         "Angular_Velocity_Non_Spherical", "Torque"]
    output_properties = []
    system_properties = list(properties_dict)

    if 'x' in system_properties:
        output_properties.append(LAMMPS_Properties[0])
    if 'xs' in system_properties:
        output_properties.append(LAMMPS_Properties[1])
    if 'xu' in system_properties:
        output_properties.append(LAMMPS_Properties[2])
    if 'xsu' in system_properties:
        output_properties.append(LAMMPS_Properties[3])
    if 'vx' in system_properties:
        output_properties.append(LAMMPS_Properties[4])
    if 'fx' in system_properties:
        output_properties.append(LAMMPS_Properties[5])
    if 'ix' in system_properties:
        output_properties.append(LAMMPS_Properties[6])
    if 'mux' in system_properties:
        output_properties.append(LAMMPS_Properties[7])
    if 'omegax' in system_properties:
        output_properties.append(LAMMPS_Properties[8])
    if 'angmomx' in system_properties:
        output_properties.append(LAMMPS_Properties[9])
    if 'tqx' in system_properties:
        output_properties.append(LAMMPS_Properties[10])

    return output_properties

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


