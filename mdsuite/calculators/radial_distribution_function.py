"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.
"""

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
import logging
from abc import ABC

import numpy as np
import matplotlib.pyplot as plt
import warnings

# Import user packages
from tqdm import tqdm
import tensorflow as tf
import itertools
from mdsuite.utils.meta_functions import join_path

from mdsuite.calculators.calculator import Calculator
from mdsuite.utils.meta_functions import split_array

from timeit import default_timer as timer

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")

log = logging.getLogger(__file__)


class RadialDistributionFunction(Calculator, ABC):
    """
    Class for the calculation of the radial distribution function

    Attributes
    ----------
    experiment :  object
            Experiment class to call from
    plot : bool
            if true, plot the tensor_values
    data_range :
            Number of configurations to use in each ensemble
    save :
            If true, tensor_values will be saved after the analysis
    x_label : str
            X label of the tensor_values when plotted
    y_label : str
            Y label of the tensor_values when plotted
    analysis_name : str
            Name of the analysis
    loaded_property : str
            Property loaded from the database_path for the analysis
    minibatch: int, default None
            Size of a individual minibatch, if set. By default minibatching is not applied

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.AngularDistributionFunction(number_of_configurations = 500, minibatch = 0, start = 0,
                                                           stop = 1000, number_of_bins = 100, use_tf_function = False)
    """

    def __init__(self, experiment):
        # TODO move all arguments from the __init__ to __call__ except for those, that are independent of user input,
        #  e.g. experiment
        """

        Attributes
        ----------
        experiment :  object
                Experiment class to call from
        """
        super().__init__(experiment)
        self.experiment = experiment

        self.scale_function = {'quadratic': {'outer_scale_factor': 1}}

        self.loaded_property = 'Positions'  # Which database_path property to load

        self.database_group = 'Radial_Distribution_Function'  # Which database_path group to save the tensor_values in
        self.x_label = r'r ($\AA$)'
        self.y_label = 'g(r)'
        self.analysis_name = 'Radial_Distribution_Function'
        # self.system_property = "RDF"
        self.experimental = True

        # Arguments set by the user in __call__
        self.number_of_bins = None  # Number of number_of_bins to use in the histogram
        self.cutoff = None  # Cutoff for the RDF
        self.images = None  # number of images to include
        self.start = None  # Which configuration to start at
        self.stop = None  # Which configuration to stop at
        self.number_of_configurations = None  # Number of configurations to use
        self.molecules = None
        self.minibatch = None
        self.use_tf_function = None

        self.override_n_batches = None

        # variables generated from user input in __call__
        self.index_list = None
        self.bin_range = [0, self.cutoff]  # set the bin range
        self.sample_configurations = None
        self.key_list = None  # Select combinations
        self.rdf = None  # instantiate the rdf tuples

    def __call__(self, plot=True, number_of_bins=None, cutoff=None, save=True, data_range=1,
                 images=1, start=0, stop=None, number_of_configurations=500, export: bool = False,
                 minibatch: int = -1, molecules: bool = False, gpu: bool = False, **kwargs):
        """Compute the RDF with the given user parameters

        Parameters
        ----------
        plot: bool
            Plot the RDF after the computation
        number_of_bins: int
            The number of bins for the RDF histogram
        cutoff: float
            The cutoff value for the RDF. Default is half the box size
        save: bool
            save the data
        data_range: int
            TODO
        images # TODO
        start: int
            Starting position in the database. All values before start will be ignored.
        stop: int
            Stopping position in the database. All values after stop will be ignored.
        number_of_configurations: int
            The number of uniformly sampled configuration between start and stop to be used for the RDF.
        export: bool
            TODO
        minibatch: int
            Size of a minibatch over atoms in the batch over configurations. Decrease this value if you run into memory
            issues. Increase this value for better performance.
        molecules: bool
            TODO
        gpu: bool
            Calculate batch size based on GPU memory instead of CPU memory
        kwargs:
            override_n_batches: int - override the automatic batch size calculation

        Returns
        -------

        """
        self.update_user_args(plot=plot, save=save, data_range=data_range, export=export, gpu=gpu)
        # User Arguments
        self.number_of_bins = number_of_bins  # Number of number_of_bins to use in the histogram
        self.cutoff = cutoff  # Cutoff for the RDF
        self.images = images  # number of images to include
        self.start = start  # Which configuration to start at
        self.stop = stop  # Which configuration to stop at
        self.number_of_configurations = number_of_configurations  # Number of configurations to use
        self.molecules = molecules
        self.use_tf_function = kwargs.pop("use_tf_function", False)

        self.override_n_batches = kwargs.get('batches')

        # Perform checks
        if stop is None:
            self.stop = self.experiment.number_of_configurations - 1

        if self.cutoff is None:
            self.cutoff = self.experiment.box_array[0] / 2  # set cutoff to half box size if none set

        if number_of_configurations == -1:
            self.number_of_configurations = self.experiment.number_of_configurations - 1

        if minibatch == -1:
            self.minibatch = self.number_of_configurations
        else:
            self.minibatch = minibatch

        if self.number_of_bins is None:
            self.number_of_bins = int(self.cutoff / 0.01)  # default is 1/100th of an angstrom

        # Set calculation specific parameters
        self.bin_range = [0, self.cutoff]  # set the bin range
        if self.molecules:
            self.index_list = [i for i in range(len(self.experiment.molecules.keys()))]
            # Get the indices of the species
        else:
            self.index_list = [i for i in range(len(self.experiment.species.keys()))]  # Get the indices of the species
        self.sample_configurations = np.linspace(self.start,
                                                 self.stop,
                                                 self.number_of_configurations,
                                                 dtype=np.int)  # choose sampled configurations
        self.key_list = [self._get_species_names(x) for x in
                         list(itertools.combinations_with_replacement(self.index_list, r=2))]  # Select combinations
        self.rdf = {name: np.zeros(self.number_of_bins) for name in self.key_list}  # instantiate the rdf tuples

        out = self.run_analysis()

        # TODO auto number_of_configurations  / bin_range

        self.experiment.save_class()
        # need to move save_class() to here, because it can't be done in the experiment any more!

        return out

    def _get_ideal_gas_probability(self) -> float:
        """
        Get the correct ideal gas term

        In the case of a cutoff value greater than half of the box size, the ideal gas term of the experiment must be
        corrected due to the lack of spherical symmetry in the experiment.

        Returns
        -------
        correction : float
                Correct ideal gas term for the RDF prefactor
        """

        def _spherical_symmetry(data: np.array) -> np.array:
            """
            Operation to perform for full spherical symmetry

            Parameters
            ----------
            data : np.array
                    tensor_values on which to operate
            Returns
            -------
            function_values : np.array
                    result of the operation
            """
            return 4 * np.pi * (data ** 2)

        def _correction_1(data: np.array) -> np.array:
            """
            First correction to ideal gas.

            tensor_values : np.array
                    tensor_values on which to operate
            Returns
            -------
            function_values : np.array
                    result of the operation

            """

            return 2 * np.pi * data * (3 - 4 * data)

        def _correction_2(data: np.array) -> np.array:
            """
            Second correction to ideal gas.

            tensor_values : np.array
                    tensor_values on which to operate
            Returns
            -------
            function_values : np.array
                    result of the operation

            """

            arctan_1 = np.arctan(np.sqrt(4 * (data ** 2) - 1))
            arctan_2 = 8 * data * np.arctan(
                (2 * data * (4 * (data ** 2) - 3)) / (np.sqrt(4 * (data ** 2) - 2) * (4 * (data ** 2) + 1)))
            return 2 * data * (3 * np.pi - 12 * arctan_1 + arctan_2)

        def _piecewise(data: np.array) -> np.array:
            """
            Return a piecewise operation on a set of tensor_values
            Parameters
            ----------
            data : np.array
                    tensor_values on which to operate

            Returns
            -------
            scaled_data : np.array
                    tensor_values that has been operated on.
            """

            # Boundaries on the ideal gsa correction. These go to 73% over half the box size, the most for a cubic box.
            lower_bound = self.experiment.box_array[0] / 2
            middle_bound = np.sqrt(2) * self.experiment.box_array[0] / 2

            # split the tensor_values into parts
            split_1 = split_array(data, data <= lower_bound)
            if len(split_1) == 1:
                return _spherical_symmetry(split_1[0])
            else:
                split_2 = split_array(split_1[1], split_1[1] < middle_bound)
                if len(split_2) == 1:
                    return np.concatenate((_spherical_symmetry(split_1[0]), _correction_1(split_2[0])))
                else:
                    return np.concatenate((_spherical_symmetry(split_1[0]),
                                           _correction_1(split_2[0]),
                                           _correction_2(split_2[1])))

        bin_width = self.cutoff / self.number_of_bins
        bin_edges = np.linspace(0.0, self.cutoff, self.number_of_bins)

        return bin_width * _piecewise(np.array(bin_edges))

    def _load_positions(self, indices: list) -> tf.Tensor:
        """
        Load the positions matrix

        This function is here to optimize calculation speed

        Parameters
        ----------
        indices : list
                List of indices to take from the database_path
        Returns
        -------
        loaded_data : tf.Tensor
                tf.Tensor of tensor_values loaded from the hdf5 database_path
        """

        if self.molecules:
            path_list = [join_path(species, "Positions") for species in self.experiment.molecules]
            return self.experiment.load_matrix("Positions", path=path_list, select_slice=np.s_[:, indices])
        else:
            return self.experiment.load_matrix("Positions", select_slice=np.s_[:, indices])

    def _get_species_names(self, species_tuple: tuple) -> str:
        """ Get the correct names of the species being studied

        Parameters
        ----------
        species_tuple : tuple
                The species tuple i.e (1, 2) corresponding to the rdf being calculated

        Returns
        -------
        names : str
                Prefix for the saved file
        """

        if self.molecules:
            species = list(self.experiment.molecules)  # load all of the species
        else:
            species = list(self.experiment.species)  # load all of the species

        return f"{species[species_tuple[0]]}_{species[species_tuple[1]]}"

    @staticmethod
    def get_neighbour_list(positions: tf.Tensor, cell: list = None, batch_size: int = None) -> tf.Tensor:
        """
        Generate the neighbour list

        Parameters
        ----------
        positions: tf.Tensor
            Tensor with shape (number_of_configurations, n_atoms, 3) representing the coordinates
        cell: list
            If periodic boundary conditions are used, please supply the cell dimensions, e.g. [13.97, 13.97, 13.97].
            If the cell is provided minimum image convention will be applied!
        batch_size: int
            Has to be evenly divisible by the the number of configurations.

        Returns
        -------
        generator object which results all distances for the current batch of time steps

        To get the real r_ij matrix for one time_step you can use the following:
            r_ij_mat = np.zeros((n_atoms, n_atoms, 3))
            r_ij_mat[np.triu_indices(n_atoms, k = 1)] = get_neighbour_list(``*args``)
            r_ij_mat -= r_ij_mat.transpose(1, 0, 2)

        """

        def get_triu_indicies(n_atoms: int) -> tf.Tensor:
            """
            Version of np.triu_indices with k=1

            Parameters
            ----------
            n_atoms : int
                    Number of atoms to perform operation on
            Returns
            ---------
                Returns a vector of size (2, None) instead of a tuple of two values like np.triu_indices
            """
            bool_mat = tf.ones((n_atoms, n_atoms), dtype=tf.bool)
            # Just construct a boolean true matrix the size of one time_step
            indices = tf.where(~tf.linalg.band_part(bool_mat, -1, 0))
            # Get the indices of the lower triangle (without diagonals)
            indices = tf.cast(indices, dtype=tf.int32)  # Get the correct dtype
            return tf.transpose(indices)  # Return the transpose for convenience later

        def get_rij_mat(positions: tf.Tensor, triu_mask: tf.Tensor, cell: list) -> tf.Tensor:
            """
            Use the upper triangle of the virtual r_ij matrix constructed of n_atoms * n_atoms matrix and subtract
            the transpose to get all distances once! If PBC are used, apply the minimum image convention.

            Parameters
            ----------
            positions : tf.Tensor
            triu_mask : tf.Tensor
            cell : list
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
    def _apply_system_cutoff(tensor: tf.Tensor, cutoff: float) -> tf.Tensor:
        """
        Enforce a cutoff on a tensor

        Parameters
        ----------
        tensor : tf.Tensor
        cutoff : flaot
        """

        cutoff_mask = tf.cast(tf.less(tensor, cutoff), dtype=tf.bool)  # Construct the mask

        return tf.boolean_mask(tensor, cutoff_mask)

    @staticmethod
    def _bin_data(distance_tensor: tf.Tensor, bin_range: list = None, nbins: int = 500) -> tf.Tensor:
        """
        Build the histogram number_of_bins for the neighbour lists

        Parameters
        ----------
        distance_tensor : tf.Tensor
                Distance tensor on which the operation should be performed
        bin_range : list
                Range over which bins should be constructed
        nbins : int
                number of bins to include in the histogram
        """

        if bin_range is None:
            bin_range = [0.0, 5.0]

        return tf.histogram_fixed_width(distance_tensor, bin_range, nbins)

    def get_pair_indices(self, len_elements: list, index_list: list) -> tuple:
        """
        Get the indicies of the pairs for rdf calculation

        Parameters
        ----------
        len_elements: list
            length of all species/elements in the simulation
        index_list: list
            list of the indices of the species

        Returns
        -------
        list, string: returns a 1D array of the positions of the pairs in the r_ij_mat, name of the pairs

        """

        n_atoms = sum(len_elements)  # Get the total number of atoms in the experiment
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

    def _calculate_histograms(self):
        """
        Build the rdf dictionary up with histogram tensor_values

        Returns
        -------
        update the class state
        """
        for i in tqdm(np.array_split(self.sample_configurations, self.n_batches), ncols=70):
            log.debug('Loading Data')
            if self.molecules:
                if len(self.experiment.molecules) == 1:
                    positions = [self._load_positions(i)]  # Load the batch of positions
                else:
                    positions = self._load_positions(i)  # Load the batch of positions
            else:
                if len(self.experiment.species) == 1:
                    positions = [self._load_positions(i)]  # Load the batch of positions
                else:
                    positions = self._load_positions(i)  # Load the batch of positions
            log.debug('Finished data loading.')

            positions_tensor = tf.concat(positions, axis=0)  # Combine all elements in one tensor
            positions_tensor = tf.transpose(positions_tensor, (1, 0, 2))  # Change to (time steps, n_atoms, coords)
            # Compute all distance vectors
            r_ij_mat = next(self.get_neighbour_list(positions_tensor, cell=self.experiment.box_array))
            for pair, names in self.get_pair_indices([len(x) for x in positions],
                                                     self.index_list):  # Iterate over all pairs
                distance_tensor = tf.norm(tf.gather(r_ij_mat, pair, axis=1), axis=2)  # Compute all distances
                distance_tensor = self._apply_system_cutoff(distance_tensor, self.cutoff)
                self.rdf[names] += np.array(self._bin_data(distance_tensor, bin_range=self.bin_range,
                                                           nbins=self.number_of_bins), dtype=float)

            log.debug('Finished RDF computation')

    def _calculate_prefactor(self, species: str) -> float:
        """
        Calculate the relevant prefactor for the analysis

        Parameters
        ----------
        species : str
                The species tuple of the RDF being studied, e.g. Na_Na
        """

        species_scale_factor = 1
        species_split = species.split("_")
        if species_split[0] == species_split[1]:
            species_scale_factor = 2

        if self.molecules:
            # Density of all atoms / total volume
            rho = len(self.experiment.molecules[species_split[1]]['indices']) / self.experiment.volume
            ideal_correction = self._get_ideal_gas_probability()  # get the ideal gas value
            numerator = species_scale_factor
            denominator = self.number_of_configurations * rho * ideal_correction * \
                          len(self.experiment.molecules[species_split[0]]['indices'])
        else:
            # Density of all atoms / total volume
            rho = len(self.experiment.species[species_split[1]]['indices']) / self.experiment.volume
            ideal_correction = self._get_ideal_gas_probability()  # get the ideal gas value
            numerator = species_scale_factor
            denominator = self.number_of_configurations * rho * ideal_correction * \
                          len(self.experiment.species[species_split[0]]['indices'])
        prefactor = numerator / denominator

        return prefactor

    def _calculate_radial_distribution_functions(self):
        """
        Take the calculated histograms and apply the correct pre-factor to them to get the correct RDF.
        Returns
        -------
        Updates the class state
        """
        for names in self.key_list:
            prefactor = self._calculate_prefactor(names)  # calculate the prefactor

            self.rdf.update({names: self.rdf.get(names) * prefactor})  # Apply the prefactor

            if self.plot:
                fig, ax = plt.subplots()
                ax.plot(np.linspace(0.0, self.cutoff, self.number_of_bins), self.rdf.get(names), label=names)
                self._plot_fig(fig, ax, title=names)  # Plot the tensor_values if necessary

            self.data_range = self.number_of_configurations
            if self.save:
                data = [{"x": x, "y": y} for x, y in
                        zip(np.linspace(0.0, self.cutoff, self.number_of_bins), self.rdf.get(names))]
                log.debug("Writing RDF to database!")
                self._update_properties_file({
                    "Property": "RDF",
                    "Analysis": self.analysis_name,
                    "subjects": names.split("_"),
                    "data_range": self.data_range,
                    "data": data
                })
            if self.export:
                self._export_data(name=self._build_table_name(names),
                                  data=self._build_pandas_dataframe(np.linspace(0.0, self.cutoff, self.number_of_bins),
                                                                    self.rdf.get(names)))

        self.experiment.radial_distribution_function_state = True  # update the state

    def run_experimental_analysis(self):
        """
        Perform the rdf analysis
        """
        log.info('Starting RDF Calculation')

        if self.batch_size > self.number_of_configurations:
            self.batch_size = self.number_of_configurations
            self.n_batches = 1
        else:
            self.n_batches = int(self.number_of_configurations / self.batch_size)

        if self.override_n_batches is not None:
            self.n_batches = self.override_n_batches

        if self.minibatch is None:
            log.debug(f"Doing full batch RDF computations")
            if self.use_tf_function:
                raise NotImplementedError('tf.function is only supported with minibatching')
            else:
                self._calculate_histograms()  # Calculate the RDFs
        else:
            log.debug(f"Using minibatching with batch size: {self.minibatch}")
            self.mini_calculate_histograms()

        self._calculate_radial_distribution_functions()

        return self.rdf

    def get_partial_triu_indices(self, n_atoms: int, m_atoms: int, idx: int) -> tf.Tensor:
        """Calculate the indices of a slice of the triu values

        Parameters
        ----------
        n_atoms: total number of atoms in the system
        m_atoms: size of the slice (horizontal)
        idx: start index of slize

        Returns
        -------
        tf.Tensor

        """
        bool_mat = tf.ones((m_atoms, n_atoms), dtype=tf.bool)
        bool_vector = ~tf.linalg.band_part(bool_mat, -1, idx)  # rename!

        indices = tf.where(bool_vector)
        indices = tf.cast(indices, dtype=tf.int32)  # is this large enough?!
        indices = tf.transpose(indices)
        return indices

    def mini_get_pair_indices(self, indices, n_atoms, atoms_per_batch, start, index_list, len_elements):
        """Similar to the pair indices, compute the mini pair indices

        Parameters
        ----------
        n_atoms: int number of atoms
        atoms_per_batch: int number of atoms in the batch
        start: start value of the slice of the batch
        index_list: list of the atoms, e.g. [0, 1, 2] for a three species system
        len_elements: list, length of the elements in the list, e.g. [50, 50, 100]

        Returns
        -------

        tf.Tensor, str

        """
        indices = tf.transpose(indices)
        background = tf.tensor_scatter_nd_update(tf.fill((atoms_per_batch, n_atoms), -1), indices,
                                                 tf.range(tf.shape(indices)[0]))

        output = []

        for tuples in itertools.combinations_with_replacement(index_list, 2):
            row_slice = (sum(len_elements[:tuples[0]]) - start, sum(len_elements[:tuples[0] + 1]) - start)
            col_slice = (sum(len_elements[:tuples[1]]), sum(len_elements[:tuples[1] + 1]))
            names = self._get_species_names(tuples)
            indices = background[slice(*row_slice), slice(*col_slice)]

            output.append([indices[indices != -1], names])

        return output

    def mini_calculate_histograms(self):
        """Do the minibatch calculation"""

        def tf_function(func):
            """Enable/Disbale tf.function based on self.use_tf_function"""
            if not self.use_tf_function:  # no use of decorator
                return func

            @tf.function
            def tf_func(*args, **kwargs):
                return func(*args, **kwargs)

            return tf_func

        @tf_function
        def compute_species_values(indices, atoms_per_batch, start, d_ij):
            rdf = {name: tf.zeros(self.number_of_bins, dtype=tf.int32) for name in self.key_list}
            for species_indices, key in self.mini_get_pair_indices(
                    indices, n_atoms, atoms_per_batch, start, self.index_list, [x.shape[0] for x in positions]):
                # iterate over the permutations between the species
                distance_between_species = tf.gather(d_ij, species_indices)
                distance_between_species = self._apply_system_cutoff(distance_between_species, self.cutoff)
                # need to do this here! otherwise indices get mixed up

                bin_data = tf.histogram_fixed_width(distance_between_species, self.bin_range, self.number_of_bins)

                rdf[key] = bin_data
            return rdf

        @tf_function
        def combine_dictionaries(dict_a, dict_b):
            """Combine two dictionaries in a tf.function call"""
            out = dict()
            for key in dict_a:
                out[key] = dict_a[key] + dict_b[key]
            return out

        @tf_function
        def run_minibatch_loop():
            """Run a minibatch loop"""
            start = 0
            stop = 0
            rdf = {name: tf.zeros(self.number_of_bins, dtype=tf.int32) for name in self.key_list}

            for atoms in per_atoms_ds.batch(self.minibatch).prefetch(tf.data.AUTOTUNE):

                atoms_per_batch = tf.shape(atoms)[0]
                stop += atoms_per_batch

                indices = self.get_partial_triu_indices(n_atoms, atoms_per_batch, start)

                # apply the mask to this, to only get the triu values and don't compute anything twice
                _positions = tf.gather(positions_tensor, indices[1], axis=0)

                # for atoms_per_batch > 1, flatten the array according to the positions
                atoms_position = tf.gather(atoms, indices[0], axis=0)

                r_ij = _positions - atoms_position

                # apply minimum image convention
                if self.experiment.box_array is not None:
                    r_ij -= tf.math.rint(r_ij / self.experiment.box_array) * self.experiment.box_array

                d_ij = tf.linalg.norm(r_ij, axis=-1)
                _rdf = compute_species_values(indices, atoms_per_batch, start, d_ij)

                rdf = combine_dictionaries(rdf, _rdf)

                start = stop
            return rdf

        execution_time = 0

        for i in tqdm(np.array_split(self.sample_configurations, self.n_batches), ncols=70):
            log.debug('Loading Data')
            if len(self.experiment.species) == 1:
                positions = [self._load_positions(i)]  # Load the batch of positions
            else:
                positions = self._load_positions(i)  # Load the batch of positions
            positions_tensor = tf.concat(positions, axis=0)  # Combine all elements in one tensor
            log.debug('Data loaded - creating dataset')
            per_atoms_ds = tf.data.Dataset.from_tensor_slices(positions_tensor)
            # create dataset of atoms from shape (n_atoms, n_timesteps, 3)

            n_atoms = tf.shape(positions_tensor)[0]
            log.debug('Starting calculation')
            start = timer()
            _rdf = run_minibatch_loop()

            for key in self.rdf:
                self.rdf[key] += _rdf[key]

            execution_time += timer() - start
            log.debug('Calculation done')

        self.rdf.update({key: np.array(val.numpy(), dtype=np.float) for key, val in self.rdf.items()})
        log.debug(f"RDF execution time: {execution_time} s")

    def _apply_operation(self, data, index):
        """
        Perform operation on an ensemble.

        Parameters
        ----------
        One tensor_values range of tensor_values to operate on.

        Returns
        -------

        """
        pass

    def _apply_averaging_factor(self):
        """
        Apply an averaging factor to the tensor_values.
        Returns
        -------
        averaged copy of the tensor_values
        """
        pass

    def _post_operation_processes(self, species: str = None):
        """
        call the post-op processes
        Returns
        -------

        """
        pass

    def _update_output_signatures(self):
        """
        After having run _prepare managers, update the output signatures.

        Returns
        -------

        """
        pass
