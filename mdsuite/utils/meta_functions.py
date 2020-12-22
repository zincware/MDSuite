"""
Author: Samuel Tovey
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com
Purpose: This file contains arbitrary functions used in several different processes. They are often generic and serve
         smaller purposes in order to clean up code in more important parts of the program.
"""

import os
from functools import wraps
from time import time

import psutil
import GPUtil
import numpy as np
from scipy.signal import savgol_filter

import mdsuite.utils.constants as constants


def get_dimensionality(box):
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


def extract_extxyz_properties(properties_dict):
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


def get_machine_properties():
    """ Get the properties of the machine being used """

    machine_properties = {}
    available_memory = psutil.virtual_memory().available  # RAM available
    total_cpu_cores = psutil.cpu_count(logical=True)  # CPU cores available
    total_gpu_devices = GPUtil.getGPUs()  # get information on all the gpu's

    # Update the machine properties dictionary
    machine_properties['cpu'] = total_cpu_cores
    machine_properties['memory'] = available_memory
    machine_properties['gpu'] = {}
    for gpu in total_gpu_devices:
        machine_properties['gpu'][gpu.id] = gpu.name

    return machine_properties

def line_counter(filename):
    """
    Count the number of lines in a file
    :param filename: (str) name of file to read
    :return: lines: (int) number of lines in the file
    """

    return sum(1 for i in open(filename, 'rb'))

def optimize_batch_size(filepath, number_of_configurations):
    """ Optimize the size of batches during initial processing

    During the database construction a batch size must be chosen in order to process the trajectories with the
    least RAM but reasonable performance.
    """

    computer_statistics = get_machine_properties()  # Get computer statistics

    batch_number = None  # Instantiate parameter for correct syntax

    file_size = os.path.getsize(filepath)  # Get the size of the file
    memory_per_configuration = file_size / number_of_configurations  # get the memory per configuration
    database_memory = 0.5 * computer_statistics['memory']  # We take 50% of the available memory
    initial_batch_number = int(database_memory / memory_per_configuration)  # trivial batch allocation

    if file_size < database_memory:
        return int(number_of_configurations)
    else:
        return int(np.floor(number_of_configurations/initial_batch_number))

def linear_fitting_function(x, a, b):
    """ Linear function for line fitting

    In many cases, namely those involving an Einstein relation, a linear curve must be fit to some data. This function
    is called by the scipy curve_fit module as the model to fit to.

    args:
        x (list) -- x data for fitting
        a (float) -- fitting parameter of the gradient
        b (float) -- fitting parameter for the y intercept
    """
    return a * x + b


def simple_file_read(filename):
    """ trivially read a file and load it into an array

    There are many occasions when a file simply must be read and dumped into a file. In these cases, we call this method
    and dump data into an array. This is NOT memory safe, and should not be used for processing large trajectory files.
    """

    data_array = []
    with open(filename, 'r+') as f:
        for line in f:
            data_array.append(line.split())

    return data_array


def timeit(f):
    """ Decorator to time the execution of a method """

    @wraps(f)
    def wrap(*args, **kw):
        """ Function to wrap a method and time its execution """
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func:{f.__name__} took: {(te - ts)} sec')
        return result

    return wrap


def apply_savgol_filter(data):
    """ Apply a savgol filter for function smoothing """

    return savgol_filter(data, 17, 2)


def golden_section_search(data, a, b):
    """ Perform a golden-section search for function minimums

    The Golden-section search algorithm is one of the best min-finding algorithms available and is here used to
    the minimums of functions during analysis. For example, in the evaluation of coordination numbers the minimum
    values of the radial distribution functions must be calculated in order to define the coordination.

    This implementation will return an interval in which the minimum should exists, and does so for all of the minimums
    on the function.

    Arguments
    ---------
    data (np.array) -- data on which to find minimums
    """

    # Define the golden ratio identities
    phi_a = 1 / constants.golden_ratio
    phi_b = 1 / (constants.golden_ratio ** 2)

    h = a - b
    number_of_steps = int(np.ceil(np.log(1e-5 / h)) / np.log(phi_a))

    c = min(data[0], key=lambda x: abs(x - a + phi_b * h))
    d = min(data[0], key=lambda x: abs(x - a + phi_a * h))

    fc = data[1][np.where(data[0] == c)]
    fd = data[1][np.where(data[0] == d)]

    for k in range(number_of_steps - 1):
        if fc < fd:
            b = d
            d = c
            fd = fc
            h = phi_a * h
            c = min(data[0], key=lambda x: abs(x - a + phi_b * h))
            fc = data[1][np.where(data[0] == c)]
        else:
            a = c
            c = d
            fc = fd
            h = phi_a * h
            d = min(data[0], key=lambda x: abs(x - a + phi_a * h))
            fd = data[1][np.where(data[0] == d)]

    if fc < fd:
        return a, d
    else:
        return c, b
