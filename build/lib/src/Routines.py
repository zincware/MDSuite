"""
Author: Samuel Tovey
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Purpose: Routines required for the methods to run
"""

import numpy as np
import os
from Meta_Functions import *


def Analysis_Initialization():
    pass


def Green_Kubo_Diffusion_Coefficeints():
    pass


def Nernst_Einstein_Conductivity():
    pass


def Green_Kubo_Conductivity():
    pass


def Einstein_Helfand_Conductivity():
    pass


def Write_XYZ_File(attribute: str, analysis_name: str) -> object:
    """ Write an xyz file

    Implementation of the xyz function on arbitrary data

    args:
        attribute (str) -- Property to be written
    """

    os.chdir('{0}_Analysis'.format(analysis_name))  # Change into analysis directory

    species_list = ['Na', 'Cl'] # Hard coded until I sort my shit out
    data_matrix = [] # initialize empty matrix with minimal effort

    # Load the data
    for species in species_list:
        data_matrix.append(np.load('{0}_{1}.npy'.format(species, attribute)))

    # Write the array
    Write_XYZ(data_matrix)









