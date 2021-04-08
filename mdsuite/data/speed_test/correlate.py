"""
Module to test and assess the comptatioanl complexity of correlate functions
"""

import tensorflow as tf
import numpy as np
from scipy import signal
import time
import matplotlib.pyplot as plt

def _build_data(shape=(100, 100, 3)):
    """
    Build the test data
    """
    return tf.random.uniform(shape=shape)

def _numpy_correlate(data, N):
    """
    Perform numpy correlation on the data
    """
    vacf = np.zeros(N)
    start = time.time()
    for item in data:
        vacf += sum(np.correlate(item[:, idx], item[:, idx] for idx in [0, 1, 2])
    stop = time.time()

    return stop - start

def _scipy_correlate(data, N):
    """
    Perform numpy correlation on the data
    """
    vacf = np.zeros(N)
    start = time.time()
    for item in data:
        vacf += sum(scipy.correlate(item[:, idx], item[:, idx] for idx in [0, 1, 2])
    stop = time.time()
    
    return stop - start

def main():
    """
    Call functions and run
    """


