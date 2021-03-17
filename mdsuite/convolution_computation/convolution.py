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


def convolution(loop_range, flux, data_range, time, correlation_time):
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
    averaged_jacf = np.zeros(data_range)

    # main loop for computation
    for i in tqdm(range(loop_range), ncols=70):
        # calculate the autocorrelation
        start = i * correlation_time
        stop = start + data_range
        acf = (signal.correlate(flux[:, 0][start:stop],
                                 flux[:, 0][start:stop],
                                 mode='full', method='auto') +
                signal.correlate(flux[:, 1][start:stop],
                                 flux[:, 1][start:stop],
                                 mode='full', method='auto') +
                signal.correlate(flux[:, 2][start:stop],
                                 flux[:, 2][start:stop],
                                 mode='full', method='auto'))

        # Cut off the second half of the acf
        acf = acf[int((len(acf) / 2)):]

        integral = np.trapz(acf, x=time)
        sigma[i] = integral
        averaged_jacf += acf

    return sigma, averaged_jacf

