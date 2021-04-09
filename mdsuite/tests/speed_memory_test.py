"""
Module for the speed and memory tests of different methods.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from memory_profiler import memory_usage
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.signal import correlate
from tqdm import tqdm


def correlation():
    """
    Test the speed of the autocorrelation methods.

    Data stored as [n_atoms, data_range, time]
    """
    test_range = [10, 100, 200, 300, 500, 700, 1000, 2000, 4000, 6000, 10000, 12000, 15000, 20000, 50000]
    scaling = []

    def correlation_function(data):
        """
        Perform correlation
        """
        acf = np.zeros(2 * len(data[0]) - 1)
        for item in data:
            acf += sum([correlate(item[:, idx], item[:, idx], mode="full", method='fft') for idx in range(3)])
        return acf

    for traj_length in tqdm(test_range):
        for atom_number in test_range:
            data = tf.random.uniform((atom_number, traj_length, 3))
            start = time.time()
            correlation_function(data)
            time_taken = time.time() - start
            scaling.append([atom_number, traj_length, time_taken])

    scaling_data = np.array(scaling)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(scaling_data[:, 0], scaling_data[:, 1], scaling_data[:, 2])
    ax.set_xlabel('Number of atoms')
    ax.set_ylabel('Data range')
    ax.set_zlabel('Time (s)')
    plt.show()
    np.save('correlation_usage.npy', scaling_data)


def main():
    """
    Run all methods.
    """
    correlation()


if __name__ == "__main__":
    main()