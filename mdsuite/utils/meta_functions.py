"""
Author: Samuel Tovey
Affiliation: Institute for Computational Physics, University of Stuttgart
Contact: stovey@icp.uni-stuttgart.de ; tovey.samuel@gmail.com

Summary
-------
This file contains arbitrary functions used in several different processes. They are often generic and serve
smaller purposes in order to clean up code in more important parts of the program.
"""

import os
from functools import wraps
from time import time

import psutil
import GPUtil
import numpy as np
from scipy.signal import savgol_filter

from mdsuite.utils.units import golden_ratio
from mdsuite.utils.exceptions import NoGPUInSystem


def get_dimensionality(box):
    """
    Calculate the dimensionality of the system box

    Parameters
    ----------
    box : list
            box array of the system of the form [x, y, z]

    Returns
    -------
    dimensions : int
            dimension of the box i.e, 1 or 2 or 3 (Higher dimensions probably don't make sense just yet)
    """

    # Check if the x, y, or z entries are empty, i.e. 2 dimensions
    if box[0] == 0 or box[1] == 0 or box[2] == 0:
        dimensions = 2

    # Check if only one of the entries is non-zero, i.e. 1 dimension.
    elif box[0] == 0 and box[1] == 0 or box[0] == 0 and box[2] == 0 or box[1] == 0 and box[2] == 0:
        dimensions = 1

    # Other option is 3 dimensions.
    else:
        dimensions = 3

    return dimensions


def get_machine_properties():
    """
    Get the properties of the machine being used
    """

    machine_properties = {}
    available_memory = psutil.virtual_memory().available  # RAM available
    total_cpu_cores = psutil.cpu_count(logical=True)  # CPU cores available
    # Update the machine properties dictionary
    machine_properties['cpu'] = total_cpu_cores
    machine_properties['memory'] = available_memory
    machine_properties['gpu'] = {}

    try:
        total_gpu_devices = GPUtil.getGPUs()  # get information on all the gpu's
        for gpu in total_gpu_devices:
            machine_properties['gpu'][gpu.id] = gpu.name
    except NoGPUInSystem:
        raise NoGPUInSystem

    return machine_properties


def line_counter(filename):
    """
    Count the number of lines in a file

    This function used a memory safe method to count the number of lines in the file. Using the other data collected
    during the trajectory analysis, this is enough information to completely characterize the system.

    Parameters
    ----------
    filename : str
            Name of the file to be read in.

    Returns
    -------
    lines : int
            Number of lines in the file
    """

    return sum(1 for _ in open(filename, 'rb'))


def optimize_batch_size(filepath, number_of_configurations):
    """
    Optimize the size of batches during initial processing

    During the database construction a batch size must be chosen in order to process the trajectories with the
    least RAM but reasonable performance.

    Parameters
    ----------
    filepath : str
            Path to the file be read in. This is not opened during the process, it is simply needed to read the file
            size.

    number_of_configurations : int
            Number of configurations in the trajectory.

    Returns
    -------
    batch size : int
            Number of configurations to load in each batch
    """

    computer_statistics = get_machine_properties()  # Get computer statistics

    file_size = os.path.getsize(filepath)  # Get the size of the file
    memory_per_configuration = file_size / number_of_configurations  # get the memory per configuration
    database_memory = 0.5 * computer_statistics['memory']  # We take 50% of the available memory
    initial_batch_number = int(database_memory / (5*memory_per_configuration))  # trivial batch allocation

    # The database generation expands memory by ~5x the read in data size, accommodate this in batch size calculation.
    if 10 * file_size < database_memory:
        return int(number_of_configurations)
    else:
        return initial_batch_number


def linear_fitting_function(x, a, b):
    """
    Linear function for line fitting

    In many cases, namely those involving an Einstein relation, a linear curve must be fit to some data. This function
    is called by the scipy curve_fit module as the model to fit to.

    Parameters
    ----------
    x : np.array
            x data for fitting
    a : float
            Fitting parameter of the gradient
    b : float
            Fitting parameter for the y intercept

    Returns
    -------
    a*x + b : float
            Returns the evaluation of a linear function.
    """
    return a * x + b


def simple_file_read(filename):
    """
    Trivially read a file and load it into an array

    There are many occasions when a file simply must be read and dumped into a file. In these cases, we call this method
    and dump data into an array. This is NOT memory safe, and should not be used for processing large trajectory files.

    Parameters
    ----------
    filename : str
            Name of the file to be read in.

    Returns
    -------
    data_array: list
            Data read in by the function.
    """

    data_array = []  # define empty data array
    with open(filename, 'r+') as f:  # Open the file for reading
        for line in f:  # Loop over the lines
            data_array.append(line.split())  # Split the lines by whitespace and add to data array

    return data_array


def timeit(f):
    """
    Decorator to time the execution of a method.

    Parameters
    ----------
    f : function
            Function to be wrapped.

    Returns
    -------
    wrap : python decorator
    """

    @wraps(f)
    def wrap(*args, **kw):
        """ Function to wrap a method and time its execution """
        ts = time()  # get the initial time
        result = f(*args, **kw)  # run the function.
        te = time()  # get the time after the function as run.
        print(f'func:{f.__name__} took: {(te - ts)} sec')  # print the outcome.

        return result

    return wrap


def apply_savgol_filter(data):
    """
    Apply a savgol filter for function smoothing

    This function will simply call the scipy SavGol implementation with preset parameters for the polynomial number
    and window size.

    Parameters
    ----------
    data : list
            Array of data to be analysed.

    Returns
    -------
    filtered data : list
            Returns the filtered data directly from the scipy SavGol filter.
    """

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
    data : np.array
            Data on which to find minimums.
    a : float
            upper bound on the min finding range.
    b : float
            lower bound on the min finding range.

    Returns
    -------
    minimum range : tuple
            Returns two radii values within which the minimum can be found.
    """

    # Define the golden ratio identities
    phi_a = 1 / golden_ratio
    phi_b = 1 / (golden_ratio ** 2)

    # Get the initial range and caluclate the maximum number of steps to be performed.
    h = a - b
    number_of_steps = int(np.ceil(np.log(1e-5 / h)) / np.log(phi_a))

    # get the minimum jump in radii
    c = min(data[0], key=lambda x: abs(x - a + phi_b * h))
    d = min(data[0], key=lambda x: abs(x - a + phi_a * h))

    # Calculate the function values at this point
    fc = data[1][np.where(data[0] == c)]
    fd = data[1][np.where(data[0] == d)]

    # Perform the search
    for k in range(number_of_steps - 1):
        # Check for the smaller range and update the current interval.
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


def round_down(a, b):
    """
    Function to get the nearest lower divisor.

    If b%a is not 0, this method may be called to get the nearest number to a that makes b%a zero.

    Parameters
    ----------
    a : int
            divisor
    b : int
            target number

    Returns
    -------
    divisor : int
            nearest number to a that divides into b evenly.
    """

    remainder = 1  # initialize a remainder
    while remainder != 0:
        remainder = b % a
        a -= 1

    return a

def split_array(data: list, condition):
    """
    split an array by a condition
    Parameters
    ----------
    data : list
            data to split
    condition : unsure
            condition on which to split by

    Returns
    -------
    split_array : list
            A list of split up arrays.

    Examples
    --------
    >>> a = np.array([1, 2, 3, 10, 20, 30])
    >>> split_array(a, a < 10)
    >>> [np.array([1, 2, 3]), np.array([10, 20, 30])]

    """

    initial_split = [data[condition], data[~condition]]  # attempt to split the array

    if len(initial_split[1]) == 0:  # if the condition is never met, return only the raw data
        return [data[condition]]
    else:  # else return the whole array
        return initial_split
