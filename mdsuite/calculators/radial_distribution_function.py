"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Class for the calculation of the radial distribution function.

Author: Samuel Tovey, Fabian Zills

Summary
-------
This module contains the code for the radial distribution function. This class is called by
the Experiment class and instantiated when the user calls the Experiment.radial_distribution_function method.
The methods in class can then be called by the Experiment.radial_distribution_function method and all necessary
calculations performed.
"""
from __future__ import annotations
import logging
from abc import ABC
from typing import Union
import numpy as np
import warnings

# Import user packages
from tqdm import tqdm
import tensorflow as tf
import itertools
from mdsuite.utils.meta_functions import join_path

from mdsuite.calculators.calculator import Calculator, call
from mdsuite.database.calculator_database import Parameters
from mdsuite.utils.meta_functions import split_array
from mdsuite.utils.linalg import apply_minimum_image, get_partial_triu_indices, apply_system_cutoff

from timeit import default_timer as timer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdsuite import Experiment

# Set style preferences, turn off warning, and suppress the duplication of loading bars.
tqdm.monitor_interval = 0
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)


class RadialDistributionFunction(Calculator, ABC):
    """
    Class for the calculation of the radial distribution function

    Attributes
    ----------
    experiment :  object
            Experiment class to call from
    data_range :
            Number of configurations to use in each ensemble
    x_label : str
            X label of the tensor_values when plotted
    y_label : str
            Y label of the tensor_values when plotted
    analysis_name : str
            Name of the analysis
    loaded_property : str
            Property loaded from the database_path for the analysis
    minibatch: int, default None
            Size of a individual minibatch, if set. By default mini-batching is not applied

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.RadialDistributionFunction(number_of_configurations = 500, minibatch = 10, start = 0,
                                                           stop = 1000, number_of_bins = 100, use_tf_function = False)
    """

    def __init__(self, **kwargs):
        """
        Constructor for the RDF calculator.

        Attributes
        ----------
        kwargs: see RunComputation class for all the passed arguments
        """
        super().__init__(**kwargs)

        self.scale_function = {'quadratic': {'outer_scale_factor': 1}}
        self.loaded_property = 'Positions'
        self.database_group = 'Radial_Distribution_Function'  # Which database_path group to save the tensor_values in
        self.x_label = r'$$\text{r} \AA$$'
        self.y_label = '$$\text{g(r)}$$'
        self.analysis_name = 'Radial_Distribution_Function'
        self.experimental = True

        self._dtype = tf.float32

        # Arguments set by the user in __call__
        self.number_of_bins = None
        self.cutoff = None
        self.start = None
        self.stop = None
        self.number_of_configurations = None
        self.molecules = None
        self.minibatch = None
        self.use_tf_function = None
        self.override_n_batches = None
        self.index_list = None
        self.bin_range = None
        self.sample_configurations = None
        self.key_list = None
        self.rdf = None

        self.correct_minibatch_batching = None
        # split the minibatches into equal sized chunks to use maximum computing and memory resources

    @call
    def __call__(self,
                 plot=True,
                 number_of_bins=None,
                 cutoff=None,
                 save=True,
                 data_range=1,
                 start=0,
                 stop=None,
                 number_of_configurations=500,
                 minibatch: int = -1,
                 species: list = None,
                 molecules: bool = False,
                 gpu: bool = False,
                 **kwargs):
        """
        Compute the RDF with the given user parameters

        Parameters
        ----------
        plot: bool
            Plot the RDF after the computation
        number_of_bins: int
            The number of bins for the RDF histogram
        species : list
            A list of species to study.
        cutoff: float
            The cutoff value for the RDF. Default is half the box size
        save: bool
            save the data
        data_range: int
            None, must be here for the parent classes to work.
        start: int
            Starting position in the database. All values before start will be
            ignored.
        stop: int
            Stopping position in the database. All values after stop will be
            ignored.
        number_of_configurations: int
            The number of uniformly sampled configuration between start and
            stop to be used for the RDF.
        minibatch: int
            Size of a minibatch over atoms in the batch over configurations.
            Decrease this value if you run into memory
            issues. Increase this value for better performance.
        molecules: bool
            If true, the molecules will be analyzed rather than the atoms.
        gpu: bool
            Calculate batch size based on GPU memory instead of CPU memory
        kwargs:
            batches: int
                    override the automatic batch size calculation
            use_tf_function : bool
                    If true, tf.function is used in the calculation.
        """
        self.number_of_bins = number_of_bins
        self.cutoff = cutoff
        self.start = start
        self.stop = stop
        self.number_of_configurations = number_of_configurations
        self.molecules = molecules
        self.minibatch = minibatch
        self.species = species

        self.update_user_args(plot=plot,
                              save=save,
                              data_range=data_range,
                              gpu=gpu)

        # kwarg parsing
        self.use_tf_function = kwargs.pop("use_tf_function", False)
        self.override_n_batches = kwargs.get('batches')
        self.tqdm_limit = kwargs.pop('tqdm', 10)
        # if there are more batches than in that limit it will show the batch tqdm, otherwise
        # it will show multiple minibatch tqdms

        # Initial checks and initialization.
        self._check_input()
        self._initialize_rdf_parameters()

        # # Perform analysis and save.
        # return self.run_analysis()

    def _initialize_rdf_parameters(self):
        """
        Initialize the RDF parameters.

        Returns
        -------
        Updates class attributes.
        """
        self.bin_range = [0, self.cutoff]
        self.index_list = [i for i in range(len(self.species))]  # Get the indices of the species

        self.sample_configurations = np.linspace(self.start,
                                                 self.stop,
                                                 self.number_of_configurations,
                                                 dtype=np.int)  # choose sampled configurations

        # Generate the tuples e.g ('Na', 'Cl'), ('Na', 'Na')
        self.key_list = [self._get_species_names(x) for x in
                         list(itertools.combinations_with_replacement(self.index_list, r=2))]

        self.rdf = {name: np.zeros(self.number_of_bins) for name in self.key_list}  # instantiate the rdf tuples

    def _check_input(self):
        """
        Check the input of the call method and store defaults if needed.

        Returns
        -------
        Updates class attributes if required.
        """
        if self.stop is None:
            self.stop = self.experiment.number_of_configurations - 1

        if self.cutoff is None:
            self.cutoff = self.experiment.box_array[0] / 2 - 0.1  # set cutoff to half box size if none set

        if self.number_of_configurations == -1:
            self.number_of_configurations = self.experiment.number_of_configurations - 1

        if self.minibatch == -1:
            self.minibatch = self.number_of_configurations

        if self.number_of_bins is None:
            self.number_of_bins = int(self.cutoff / 0.01)  # default is 1/100th of an angstrom

        if self.species is None:
            self.species = list(self.experiment.species)

        if self.molecules:
            self.species = list(self.experiment.molecules)

        if self.gpu:
            self.correct_minibatch_batching = 100
            # 100 seems to be a good value for most systems

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
        path_list = [join_path(species, "Positions") for species in self.species]
        data = self.experiment.load_matrix("Positions", path=path_list, select_slice=np.s_[:, indices])
        if len(self.species) == 1:
            return tf.cast(data, dtype=self.dtype)
        else:
            return tf.cast(tf.concat(data, axis=0), dtype=self.dtype)

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

        return f"{self.species[species_tuple[0]]}_{self.species[species_tuple[1]]}"

    def _calculate_prefactor(self, species: Union[str, tuple] = None):
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
            numerator = species_scale_factor
            denominator = self.number_of_configurations * rho * self.ideal_correction * \
                          len(self.experiment.molecules[species_split[0]]['indices'])
        else:
            # Density of all atoms / total volume
            rho = len(self.experiment.species[species_split[1]]['indices']) / self.experiment.volume
            numerator = species_scale_factor
            denominator = self.number_of_configurations * rho * self.ideal_correction * \
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
        for i, names in enumerate(self.key_list):
            if i == len(self.key_list) - 1:
                self.last_iteration = True
            prefactor = self._calculate_prefactor(names)  # calculate the prefactor

            self.rdf.update({names: self.rdf.get(names) * prefactor})  # Apply the prefactor
            log.debug("Writing RDF to database!")

            self.data_range = self.number_of_configurations
            data = [{"x": x, "y": y} for x, y in
                    zip(np.linspace(0.0, self.cutoff, self.number_of_bins),
                        self.rdf.get(names))]
            params = Parameters(
                Property=self.database_group,
                Analysis=self.analysis_name,
                data_range=self.data_range,
                data=data,
                Subject=names.split("_"))

            self.update_database(params)

            if self.plot:
                self.run_visualization(
                    x_data=np.linspace(0.0, self.cutoff, self.number_of_bins),
                    y_data=self.rdf.get(names),
                    title=f"{names}",
                )

        self.experiment.radial_distribution_function_state = True  # update the state

    def _correct_batch_properties(self):
        """
        We must fix the batch size parameters set by the parent class.

        Returns
        -------
        Updates the parent class.
        """
        if self.batch_size > self.number_of_configurations:
            self.batch_size = self.number_of_configurations
            self.n_batches = 1
        else:
            self.n_batches = int(self.number_of_configurations / self.batch_size)

        if self.override_n_batches is not None:
            self.n_batches = self.override_n_batches

    def mini_calculate_histograms(self):
        """Do the minibatch calculation"""

        def combine_dictionaries(dict_a, dict_b):
            """Combine two dictionaries in a tf.function call"""
            out = dict()
            for key in dict_a:
                out[key] = dict_a[key] + dict_b[key]
            return out

        def run_minibatch_loop():
            """Run a minibatch loop"""
            minibatch_start = tf.constant(0)
            stop = tf.constant(0)
            rdf = {name: tf.zeros(self.number_of_bins, dtype=tf.int32) for name in self.key_list}

            if self.correct_minibatch_batching is not None:
                # per_atoms_ds with shape (configurations, 3)
                corrected_ds = per_atoms_ds.batch(int(len(per_atoms_ds) / self.correct_minibatch_batching))
                # math.stackexchange.com/questions/107269/how-do-you-split-a-90-45-45-triangle-into-equal-area-strips
                for jdx, corrected_per_atoms_ds in tqdm(enumerate(corrected_ds), ncols=70, disable=not batch_tqm,
                                                        total=self.correct_minibatch_batching,
                                                        desc=f"Mini batch {idx + 1}/{self.n_batches}"):
                    pre_factor = np.sqrt(jdx + 1)
                    new_ds = tf.data.Dataset.from_tensor_slices(corrected_per_atoms_ds)
                    new_ds = new_ds.batch(int(pre_factor * self.minibatch))
                    log.debug(f'batch size: {int(pre_factor * self.minibatch)} ({self.minibatch} * {pre_factor})')
                    for atoms in new_ds:
                        atoms_per_batch, batch_size, _ = tf.shape(atoms)
                        stop += atoms_per_batch
                        start_time = timer()
                        indices = get_partial_triu_indices(n_atoms, atoms_per_batch, minibatch_start)
                        log.debug(f'Calculating indices took {timer() - start_time} s')

                        start_time = timer()
                        d_ij = self.get_dij(indices, positions_tensor, atoms,
                                            tf.cast(self.experiment.box_array, dtype=self.dtype))
                        exec_time = timer() - start_time
                        atom_pairs_per_second = tf.cast(tf.shape(indices)[1], dtype=self.dtype) / exec_time / 10 ** 6
                        atom_pairs_per_second *= tf.cast(batch_size, dtype=self.dtype)
                        log.debug(f'Computing d_ij took {exec_time} s '
                                  f'({atom_pairs_per_second:.1f} million atom pairs / s)')

                        start_time = timer()
                        minibatch_rdf = self.compute_species_values(indices, minibatch_start, d_ij)
                        log.debug(f'Computing species values took {timer() - start_time} s')

                        start_time = timer()
                        rdf = combine_dictionaries(rdf, minibatch_rdf)
                        log.debug(f'Updating dictionaries took {timer() - start_time} s')

                        minibatch_start = stop

                return rdf

            else:
                for atoms in tqdm(per_atoms_ds.batch(self.minibatch).prefetch(tf.data.AUTOTUNE), ncols=70,
                                  disable=not batch_tqm, desc=f"Running mini batch loop {idx + 1} / {self.n_batches}"):
                    # I assume this code can be removed, because the corrected version is almost always better
                    # If one still wants to use the uncorrected version, we could make that an option, but
                    # I don't see why. The only downside of the other method is, that a new dataset is created
                    # every loop which can be cause a slow down, but I don't think it causes memory issues.
                    atoms_per_batch, batch_size, _ = tf.shape(atoms)
                    stop += atoms_per_batch
                    start_time = timer()
                    indices = get_partial_triu_indices(n_atoms, atoms_per_batch, minibatch_start)
                    log.debug(f'Calculating indices took {timer() - start_time} s')

                    start_time = timer()
                    d_ij = self.get_dij(indices, positions_tensor, atoms,
                                        tf.cast(self.experiment.box_array, dtype=self.dtype))
                    exec_time = timer() - start_time
                    atom_pairs_per_second = tf.cast(tf.shape(indices)[1], dtype=self.dtype) / exec_time / 10 ** 6
                    atom_pairs_per_second *= tf.cast(batch_size, dtype=self.dtype)
                    log.debug(f'Computing d_ij took {exec_time} s '
                              f'({atom_pairs_per_second:.1f} million atom pairs / s)')

                    start_time = timer()
                    minibatch_rdf = self.compute_species_values(indices, minibatch_start, d_ij)
                    log.debug(f'Computing species values took {timer() - start_time} s')

                    start_time = timer()
                    rdf = combine_dictionaries(rdf, minibatch_rdf)
                    log.debug(f'Updating dictionaries took {timer() - start_time} s')

                    minibatch_start = stop
                return rdf

        execution_time = 0

        batch_tqm = self.tqdm_limit > self.n_batches

        for idx, sample_configuration in tqdm(enumerate(np.array_split(self.sample_configurations, self.n_batches)),
                                              ncols=70, disable=batch_tqm):
            log.debug('Loading Data')

            positions_tensor = self._load_positions(sample_configuration)
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

    def run_experimental_analysis(self):
        """
        Perform the rdf analysis
        """
        log.info('Starting RDF Calculation')
        self._correct_batch_properties()

        log.debug(f"Using minibatching with batch size: {self.minibatch}")
        self.mini_calculate_histograms()

        self._calculate_radial_distribution_functions()

    def compute_species_values(self, indices: tf.Tensor, start_batch, d_ij: tf.Tensor):
        """
        Compute species-wise histograms

        Parameters
        ----------
        indices: indices of the d_ij distances in the shape (x, 2)
        start_batch: starts from 0 and increments by atoms_per_batch every batch
        d_ij: d_ij matrix in the shape (x, batches) where x comes from the triu computation

        Returns
        -------

        """
        rdf = {name: tf.zeros(self.number_of_bins, dtype=tf.int32) for name in self.key_list}
        indices = tf.transpose(indices)

        particles_list = self.particles_list

        for tuples in itertools.combinations_with_replacement(self.index_list, 2):
            names = self._get_species_names(tuples)
            start_ = tf.concat(
                [sum(particles_list[:tuples[0]]) - start_batch, sum(particles_list[:tuples[1]])],
                axis=0
            )
            stop_ = start_ + tf.constant([particles_list[tuples[0]], particles_list[tuples[1]]])
            rdf[names] = self.bin_minibatch(start_, stop_, indices, d_ij,
                                            tf.cast(self.bin_range, dtype=self.dtype),
                                            tf.cast(self.number_of_bins, dtype=tf.int32),
                                            tf.cast(self.cutoff, dtype=self.dtype))
        return rdf

    @staticmethod
    @tf.function(experimental_relax_shapes=True)
    def bin_minibatch(start, stop, indices, d_ij, bin_range, number_of_bins, cutoff) -> tf.Tensor:
        """Compute the minibatch histogram"""

        # select the indices that are within the boundaries of the current species / molecule
        mask_1 = (indices[:, 0] > start[0]) & (indices[:, 0] < stop[0])
        mask_2 = (indices[:, 1] > start[1]) & (indices[:, 1] < stop[1])

        values_species = tf.boolean_mask(d_ij, mask_1 & mask_2, axis=0)
        values = apply_system_cutoff(values_species, cutoff)
        bin_data = tf.histogram_fixed_width(
            values=values,
            value_range=bin_range,
            nbins=number_of_bins
        )

        return bin_data

    @staticmethod
    @tf.function(experimental_relax_shapes=True)
    def get_dij(indices, positions_tensor, atoms, box_array):
        """Compute the distance matrix for the minibatch"""
        start_time = timer()
        log.debug(f'Calculating indices took {timer() - start_time} s')

        # apply the mask to this, to only get the triu values and don't compute anything twice
        start_time = timer()
        _positions = tf.gather(positions_tensor, indices[1], axis=0)
        log.debug(f'Gathering positions_tensor took {timer() - start_time} s')

        # for atoms_per_batch > 1, flatten the array according to the positions
        start_time = timer()
        atoms_position = tf.gather(atoms, indices[0], axis=0)
        log.debug(f'Gathering atoms took {timer() - start_time} s')

        start_time = timer()
        r_ij = _positions - atoms_position
        log.debug(f'Computing r_ij took {timer() - start_time} s')

        # apply minimum image convention
        start_time = timer()
        if box_array is not None:
            r_ij = apply_minimum_image(r_ij, box_array)
        log.debug(f'Applying minimum image convention took {timer() - start_time} s')

        start_time = timer()
        d_ij = tf.linalg.norm(r_ij, axis=-1)
        log.debug(f'Computing d_ij took {timer() - start_time} s')

        return d_ij

    @property
    def particles_list(self):
        if self.molecules:
            particles_list = [len(self.experiment.molecules[item]['indices']) for item in self.experiment.molecules]
        else:
            particles_list = [len(self.experiment.species[item]['indices']) for item in self.experiment.species]

        return particles_list

    @property
    def ideal_correction(self) -> float:
        """
        Get the correct ideal gas term

        In the case of a cutoff value greater than half of the box size, the ideal gas term of the experiment must be
        corrected due to the lack of spherical symmetry in the experiment.

        Returns
        -------
        correction : float
                Correct ideal gas term for the RDF prefactor
        """

        # TODO make it a property
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
            arctan_1 = np.arctan(np.sqrt(4 * (data ** 2) - 2))
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
            split_1 = list(split_array(data, data <= lower_bound))
            if len(split_1) == 1:
                return _spherical_symmetry(split_1[0])
            else:
                split_2 = list(split_array(split_1[1], split_1[1] < middle_bound))
                if len(split_2) == 1:
                    return np.concatenate((_spherical_symmetry(split_1[0]), _correction_1(split_2[0])))
                else:
                    return np.concatenate((_spherical_symmetry(split_1[0]),
                                           _correction_1(split_2[0]),
                                           _correction_2(split_2[1])))

        bin_width = self.cutoff / self.number_of_bins
        bin_edges = np.linspace(0.0, self.cutoff, self.number_of_bins)

        return _piecewise(np.array(bin_edges)) * bin_width
