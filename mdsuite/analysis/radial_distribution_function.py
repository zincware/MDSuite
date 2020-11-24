"""
Class for the calculation of the radial distribution function.

Author: Samuel Tovey

Description: This module contains the code for the radial distribution function. This class is called by
the Experiment class and instantiated when the user calls the Experiment.radial_distribution_function method.
The methods in class can then be called by the Experiment.radial_distribution_function method and all necessary
calculations performed.
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import itertools

from mdsuite.analysis.analysis import Analysis

class RadialDistributionFunction(Analysis):
    """ Class for the calculation of the radial distribution function """

    def __init__(self, obj, plot=True, bins=500, cutoff=None, save=True, data_range=500, x_label='r ($\AA$)',
                 y_label='g(r)', analysis_name='radial_distribution_function'):
        """ Standard python constructor """
        super().__init__(obj,plot, save, data_range, x_label, y_label, analysis_name)
        self.parent = obj
        self.bins = bins
        self.cutoff = cutoff

        if self.cutoff is None:
            self.cutoff = self.parent.box_array[0]/2  # set cutoff to half box size if no set

    def _autocorrelation_time(self):
        """ Calculate the position autocorrelation time of the system """
        raise NotImplementedError

    @staticmethod
    @tf.function
    def _get_radial_distance(tensor):
        """ get the magnitude of the tensor """
        return tf.math.sqrt(tf.reduce_sum(tf.math.square(tensor), axis=-1))

    def _load_positions(self):
        """ Load the positions matrix

        This function is here to optimize calculation speed
        """

        return tf.convert_to_tensor(self.parent.load_matrix("Positions"))

    @staticmethod
    @tf.function
    def _enforce_exclusion_block(tensor):
        """ Apply a mask to remove diagonal elements for distance tensor """

        diagonal_mask = tf.cast(tensor, dtype=tf.bool)  # Construct the mask

        return tf.boolean_mask(tensor, diagonal_mask, axis=0)

    @staticmethod
    @tf.function
    def _apply_system_cutoff(tensor, cutoff):
        """ Enforce a cutoff on a tensor """

        cutoff_mask = tf.cast(tf.less(tensor, cutoff), dtype=tf.bool)  # Construct the mask

        return tf.boolean_mask(tensor, cutoff_mask)

    @tf.function
    def _build_distance_tensor(self, positions_tensor, reference_tensor):
        """ Build the neighbour list of atoms """

        distance_tensor = []

        for i in range(len(reference_tensor)):
            distance_tensor.append(self._get_radial_distance(positions_tensor - reference_tensor[i]))

        return tf.stack(distance_tensor)

    @staticmethod
    @tf.function
    def _bin_data(distance_tensor, bin_range=None, nbins=500):
        """ Build the histogram bins for the neighbour lists """

        if bin_range is None:
            bin_range = [0.0, 5.0]
        return tf.histogram_fixed_width(distance_tensor, bin_range, nbins)

    def run_analysis(self):
        """ Perform the rdf analysis """

        bin_range = [0, self.cutoff]  # set the bin range
        positions = self._load_positions()

        index_list = [i for i in range(len(positions))]  # Get the indices of the species

        for tuples in itertools.combinations_with_replacement(index_list, 2):

            reference_tensor = positions[tuples[0]]  # set the reference matrix
            positions_tensor = positions[tuples[1]]  # set the measurement tensor

            # generate the distance tensor
            distance_tensor = self._build_distance_tensor(positions_tensor, reference_tensor)

            if tuples[0] == tuples[1]:
                distance_tensor = self._enforce_exclusion_block(distance_tensor)  # apply exclusion block

            distance_tensor = self._apply_system_cutoff(distance_tensor, self.cutoff)  # enforce cutoff

            # generate the histogram
            rdf = np.array(self._bin_data(distance_tensor, bin_range=bin_range, nbins=self.bins), dtype=float)

            # Calculate the prefactor the system being studied
            bin_width = self.cutoff/self.bins
            bin_edges = (np.linspace(0.0, self.cutoff, self.bins)**2)*4*np.pi*bin_width
            rho = len(positions_tensor)/self.parent.volume
            numerator = 1
            denominator = len(reference_tensor[0])*rho*len(positions_tensor)*bin_edges
            prefactor = numerator/denominator

            rdf *= prefactor  # Apply the prefactor

            plt.plot(np.linspace(0.0, self.cutoff, self.bins), rdf)

            if self.save:
                self._save_data(f'{tuples}_{self.analysis_name}', [np.linspace(0.0, self.cutoff, self.bins), rdf])

        if self.plot:
            self._plot_data()  # Plot the data if necessary
