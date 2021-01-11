"""
Python module for the performance of the convolution operation

Summary
-------
In many analysis the autocorrelation of a property is required. Due to the repetition of the calculation, a common
method is here implemented to facilitate future performance optimization, as well as streamlining of code.
"""

from tqdm import tqdm
import numpy as np
from scipy import signal


def convolution(loop_range, flux, data_range, time):
    """
    Calculate the autocorrelation function and integral of a property.

    Parameters
    ----------
    loop_range : int
            Number of loops to perform. This is the number of ensembles that will be included in the average.
    flux : np.array
            Property on which autocorrelation is to be performed.
    data_range : int
            Size of the ensemble to be evaluated.
    time : list
            Time array, important for correct integration.

    Returns
    -------
    sigma : np.array
            A list of integral values taken from the autocorrelation. This will correspond to the property being
            measured from the autocorrelation function.
    """

    sigma = np.empty((loop_range,))  # define an empty array

    # main loop for computation
    for i in tqdm(range(loop_range)):
        # calculate the autocorrelation
        acf = (signal.correlate(flux[:, 0][i:i + data_range],
                                 flux[:, 0][i:i + data_range],
                                 mode='full', method='auto') +
                signal.correlate(flux[:, 1][i:i + data_range],
                                 flux[:, 1][i:i + data_range],
                                 mode='full', method='auto') +
                signal.correlate(flux[:, 2][i:i + data_range],
                                 flux[:, 2][i:i + data_range],
                                 mode='full', method='auto'))

        # Cut off the second half of the acf
        acf = acf[int((len(acf) / 2)):]

        integral = np.trapz(acf, x=time)
        sigma[i] = integral

    return sigma
