"""
Python module for complexity/memory scaling functions
"""

import numpy as np


def linear_scale_function(memory_usage: int, scale_factor: int = 1):
    """
    Apply a linear scaling to memory usage.

    Parameters
    ----------
    memory_usage : int
            naive memory usage, i.e., the exact memory of one configuration.
    scale_factor : int
            Scalar scaling factor for the memory usage in cases on non-size dependent inflation.
    Returns
    -------
    scaled_memory : int
            Amount of memory required per configuration loaded.
    """
    return memory_usage * scale_factor


def linearithmic_scale_function(memory_usage: int, scale_factor: int = 1):
    """
    Apply a linearithmic scaling to memory usage.

    Parameters
    ----------
    memory_usage : int
            naive memory usage, i.e., the exact memory of one configuration.
    scale_factor : int
            Scalar scaling factor for the memory usage in cases on non-size dependent inflation.
    Returns
    -------
    scaled_memory : int
            Amount of memory required per configuration loaded.
    """
    return scale_factor * memory_usage * np.log(memory_usage)


def quadratic_scale_function(memory_usage: int, inner_scale_factor: int = 1, outer_scale_factor: int = 1):
    """
    Apply a quadratic scaling to memory usage.

    Parameters
    ----------
    memory_usage : int
            naive memory usage, i.e., the exact memory of one configuration.
    inner_scale_factor : int
            Scalar scaling factor for the inner multiplication
    outer_scale_factor : int
            Scalar scaling factor for the outer multiplication
    Returns
    -------
    scaled_memory : int
            Amount of memory required per configuration loaded.
    """
    return outer_scale_factor * (memory_usage * inner_scale_factor)**2


def polynomial_scale_function(memory_usage: int, inner_scale_factor: int = 1,
                              outer_scale_factor: int = 1, order: int = 3):
    """
    Apply a polynomial scaling to memory usage.

    Parameters
    ----------
    memory_usage : int
            naive memory usage, i.e., the exact memory of one configuration.
    inner_scale_factor : int
            Scalar scaling factor for the inner multiplication
    outer_scale_factor : int
            Scalar scaling factor for the outer multiplication
    order : int
            Order of the polynomial.
    Returns
    -------
    scaled_memory : int
            Amount of memory required per configuration loaded.
    """
    return outer_scale_factor * (memory_usage * inner_scale_factor)**order
