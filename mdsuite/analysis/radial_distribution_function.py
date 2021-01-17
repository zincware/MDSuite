"""
Class for the calculation of the radial distribution function.

Author: Samuel Tovey, Fabian Zills

Summary
-------
This module contains the code for the radial distribution function. This class is called by
the Experiment class and instantiated when the user calls the Experiment.radial_distribution_function method.
The methods in class can then be called by the Experiment.radial_distribution_function method and all necessary
calculations performed.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

# Import user packages
from tqdm import tqdm
import tensorflow as tf
import itertools

from mdsuite.analysis.analysis import Analysis

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
plt.style.use('bmh')
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")


class RadialDistributionFunction(Analysis):
    """ Class for the calculation of the radial distribution function

    Attributes
    ----------
    obj :  object
            Experiment class to call from
    plot : bool
            if true, plot the data
    data_range :
            Number of configurations to use in each ensemble
    save :
            If true, data will be saved after the analysis
    x_label : str
            X label of the data when plotted
    y_label : str
            Y label of the data when plotted
    analysis_name : str
            Name of the analysis
    loaded_property : str
            Property loaded from the database for the analysis
    correlation_time : int
            Correlation time of the property being studied. This is used to ensure ensemble sampling is only performed
            on uncorrelated samples. If this is true, the error extracted form the calculation will be correct.
    """

    def __init__(self, obj, plot=True, bins=500, cutoff=None, save=True, data_range=1, x_label=r'r ($\AA$)',
                 y_label='g(r)', analysis_name='radial_distribution_function', periodic=True, images=1, start=0,
                 stop=None, n_confs=1000, n_batches=20):
        """

        Attributes
        ----------
        obj :  object
                Experiment class to call from
        plot : bool
                if true, plot the data
        data_range :
                Number of configurations to use in each ensemble
        save :
                If true, data will be saved after the analysis
        x_label : str
                X label of the data when plotted
        y_label : str
                Y label of the data when plotted
        analysis_name : str
                Name of the analysis
        """

        super().__init__(obj, plot, save, data_range, x_label, y_label, analysis_name)
        self.parent = obj
        self.bins = bins
        self.cutoff = cutoff
        self.correlation_time = None
        self.loaded_property = 'Positions'

        self.loop_range = obj.number_of_configurations - self.data_range - 1
        self.periodic = periodic  # whether or not to apply PBC
        self.images = images  # number of images to include
        self.start = start
        self.n_confs = n_confs
        if n_batches == -1:
            self.n_batches = n_confs
        else:
            self.n_batches = n_batches

        if stop is None:
            self.stop = obj.number_of_configurations - self.data_range - 1
        else:
            self.stop = stop

        if self.cutoff is None:
            self.cutoff = self.parent.box_array[0] / 2  # set cutoff to half box size if no set

    def _autocorrelation_time(self):
        """ Calculate the position autocorrelation time of the system """
        raise NotImplementedError

    def _load_positions(self, indices):
        """ Load the positions matrix

        This function is here to optimize calculation speed
        """

        return self.parent.load_matrix("Positions", select_slice=np.s_[:, indices], tensor=True)

    @staticmethod
    def get_neighbour_list(positions, cell=None, batch_size=None):
        """Generate the neighbour list

        Parameters
        ----------
        positions: tf.Tensor
            Tensor with shape (n_timesteps, n_atoms, 3) representing the coordinates
        cell: list
            If periodic boundary conditions are used, please supply the cell dimensions, e.g. [13.97, 13.97, 13.97].
            If the cell is provided minimum image convention will be applied!
        batch_size: int
            Has to be evenly divisible by the the number of timesteps.

        Returns
        -------
            generator object which results all distances for the current batch of timesteps

            To get the real r_ij matrix for one timestep you can use the following:
                r_ij_mat = np.zeros((n_atoms, n_atoms, 3))
                r_ij_mat[np.triu_indices(n_atoms, k = 1)] = get_neighbour_list(*args)
                r_ij_mat -= r_ij_mat.transpose(1, 0, 2)

        """

        def get_triu_indicies(n_atoms):
            """Version of np.triu_indices with k=1
            Returns
            ---------
                Returns a vector of size (2, None) insted of a tuple of two values like np.triu_indices
            """
            bool_mat = tf.ones((n_atoms, n_atoms), dtype=tf.bool)
            # Just construct a boolean true matrix the size of one timestep
            indices = tf.where(~tf.linalg.band_part(bool_mat, -1, 0))
            # Get the indices of the lower triangle (without diagonals)
            indices = tf.cast(indices, dtype=tf.int32)  # Get the correct dtype
            return tf.transpose(indices)  # Return the transpose for convenience later

        def get_rij_mat(positions, triu_mask, cell):
            """Use the upper triangle of the virtual r_ij matrix constructed of n_atoms * n_atoms matrix and subtract
            the transpose to get all distances once!
            If PBC are used, apply the minimum image convention.
            """
            r_ij_mat = tf.gather(positions, triu_mask[0], axis=1) - tf.gather(positions, triu_mask[1], axis=1)
            if cell:
                r_ij_mat -= tf.math.rint(r_ij_mat / cell) * cell
            return r_ij_mat

        n_atoms = positions.shape[1]
        triu_mask = get_triu_indicies(n_atoms)

        if batch_size is not None:
            try:
                assert positions.shape[0] % batch_size == 0
            except AssertionError:
                print(
                    f"positions must be evenly divisible by batch_size, but are {positions.shape[0]} and {batch_size}")

            for positions_batch in tf.split(positions, batch_size):
                yield get_rij_mat(positions_batch, triu_mask, cell)
        else:
            yield get_rij_mat(positions, triu_mask, cell)

    @staticmethod
    def _apply_system_cutoff(tensor, cutoff):
        """ Enforce a cutoff on a tensor """

        cutoff_mask = tf.cast(tf.less(tensor, cutoff), dtype=tf.bool)  # Construct the mask

        return tf.boolean_mask(tensor, cutoff_mask)

    @staticmethod
    def _bin_data(distance_tensor, bin_range=None, nbins=500):
        """ Build the histogram bins for the neighbour lists """

        if bin_range is None:
            bin_range = [0.0, 5.0]

        return tf.histogram_fixed_width(distance_tensor, bin_range, nbins)

    def _get_species_names(self, species_tuple):
        """ Get the correct names of the species being studied

        :argument species_tuple (tuple) -- The species tuple i.e (1, 2) corresponding to the rdf being calculated

        :returns names (string) -- Prefix for the saved file
        """

        species = list(self.parent.species)  # load all of the species

        return f"{species[species_tuple[0]]}_{species[species_tuple[1]]}"

    def get_pair_indices(self, len_elements, index_list):
        """Get the indicies of the pairs for rdf calculation

        Parameters
        ----------
        len_elements: list
            length of all species/elements in the simulation
        index_list: list
            list of the indices of the species

        Returns
        -------
        array, string: returns a 1D array of the positions of the pairs in the r_ij_mat, name of the pairs

        """
        n_atoms = sum(len_elements)  # Get the total number of atoms in the system
        background = np.full((n_atoms, n_atoms), -1)  # Create a full matrix filled with placeholders
        background[np.triu_indices(n_atoms, k=1)] = np.arange(len(np.triu_indices(n_atoms, k=1)[0]))
        # populate the triu with the respecting indices in the r_ij_matrix
        for tuples in itertools.combinations_with_replacement(index_list, 2):  # Iterate over pairs
            row_slice = (sum(len_elements[:tuples[0]]), sum(len_elements[:tuples[0] + 1]))
            col_slice = (sum(len_elements[:tuples[1]]), sum(len_elements[:tuples[1] + 1]))
            # Yield the slices of the pair being investigated
            names = self._get_species_names(tuples)
            indices = background[slice(*row_slice), slice(*col_slice)]  # Get the indices for the pairs in the r_ij_mat
            yield indices[indices != -1].flatten(), names  # Remove placeholders

    def _calculate_prefactor(self, species):
        """ Calculate the relevant prefactor for the analysis """

        species_scale_factor = 1
        species_split = species.split("_")
        if species_split[0] == species_split[1]:
            species_scale_factor = 2

        # Calculate the prefactor of the system being studied
        bin_width = self.cutoff / self.bins
        bin_edges = (np.linspace(0.0, self.cutoff, self.bins) ** 2) * 4 * np.pi * bin_width



        rho = len(self.parent.species[species_split[1]]['indices']) / self.parent.volume  # Density all atoms / total volume
        numerator = species_scale_factor
        denominator = self.n_confs * rho * bin_edges * len(self.parent.species[species_split[0]]['indices'])
        prefactor = numerator / denominator

        return prefactor

    def run_analysis(self):
        """
        Perform the rdf analysis
        """

        #self._collect_machine_properties()              # collect machine properties and determine batch size

        bin_range = [0, self.cutoff]  # set the bin range
        index_list = [i for i in range(len(self.parent.species.keys()))]  # Get the indices of the species e.g. [0, 1, 2, 3]

        sample_configs = np.linspace(self.start, self.stop, self.n_confs, dtype=np.int)  # choose sampled configurations

        key_list = [self._get_species_names(x) for x in list(itertools.combinations_with_replacement(index_list, r=2))]  # Select combinations of species

        rdf = {name: np.zeros(self.bins) for name in key_list}  # e.g {"Na_Cl": [0, 0, 0, 0]}

        for i in tqdm(np.array_split(sample_configs, self.n_batches), ncols=70):
            positions = self._load_positions(i)    # Load the batch of positions
            # print(positions[0][:, -1])
            tmp = tf.concat(positions, axis=0)     # Combine all elements in one tensor
            tmp = tf.transpose(tmp, (1, 0, 2))     # Change to (timesteps, n_atoms, cart)
            r_ij_mat = next(self.get_neighbour_list(tmp, cell=self.parent.box_array))  # Compute all distance vectors
            for pair, names in self.get_pair_indices([len(x) for x in positions], index_list):  # Iterate over all pairs
                distance_tensor = tf.norm(tf.gather(r_ij_mat, pair, axis=1), axis=2)            # Compute all distances
                distance_tensor = self._apply_system_cutoff(distance_tensor, self.cutoff)
                # print(names)
                rdf[names] += np.array(self._bin_data(distance_tensor, bin_range=bin_range, nbins=self.bins),
                                       dtype=float)

        for names in key_list:
            prefactor = self._calculate_prefactor(names)  # calculate the prefactor
            rdf.update({names: rdf.get(names) * prefactor})  # Apply the prefactor
            plt.plot(np.linspace(0.0, self.cutoff, self.bins), rdf.get(names))
            plt.title(names)
            plt.show()
            if self.save:  # get the species names
                self._save_data(f'{names}_{self.analysis_name}',
                                [np.linspace(0.0, self.cutoff, self.bins), rdf.get(names)])

        if self.plot:
            self._plot_data()  # Plot the data if necessary