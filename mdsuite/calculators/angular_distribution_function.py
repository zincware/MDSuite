"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.
"""
from abc import ABC
import logging
import tensorflow as tf
import itertools
import numpy as np
from tqdm import tqdm
from mdsuite.calculators.calculator import Calculator, call
from mdsuite.utils.neighbour_list import get_neighbour_list, get_triu_indicies, get_triplets
from mdsuite.utils.linalg import get_angles
from mdsuite.database.calculator_database import Parameters
from mdsuite.utils.meta_functions import join_path

log = logging.getLogger(__name__)


class AngularDistributionFunction(Calculator, ABC):
    """
    Compute the Angular Distribution Function for all species combinations

    Attributes
    ----------
    batch_size : int
        Number of batches, to split the configurations into.
    n_minibatches: int
        Number of minibatches for computing the angles to split each batch into
    n_confs: int
        Number of configurations to analyse
    r_cut: float
        cutoff radius for the ADF
    start: int
        Index of the first configuration
    stop: int
        Index of the last configuration
    bins: int
        bins for the ADF
    use_tf_function: bool, default False
        activate the tf.function decorator for the minibatches. Can speed up the calculation significantly, but
        may lead to excessive use of memory! During the first batch, this function will be traced. Tracing is slow,
        so this might only be useful for a larger number of batches.

    See Also
    --------
    mdsuite.calculators.calculator.Calculator class

    Examples
    --------
    experiment.run_computation.AngularDistributionFunction(n_confs = 100,
                                                           r_cut = 3.2,
                                                           batch_size = 10,
                                                           n_minibatches = 50,
                                                           start = 0,
                                                           stop = 200,
                                                           bins = 100,
                                                           use_tf_function = False)
    """

    def __init__(self, **kwargs):
        """
        Compute the Angular Distribution Function for all species combinations

        Parameters
        ----------
        experiment : object
                Experiment object from which to take attributes.
        """
        super().__init__(**kwargs)
        self.scale_function = {'quadratic': {'outer_scale_factor': 10}}
        self.loaded_property = 'Positions'

        self.use_tf_function = None
        self.r_cut = None
        self.start = None
        self.molecules = None
        self.experimental = True
        self.stop = None
        self.n_confs = None
        self.bins = None
        self.bin_range = None
        self._batch_size = None  # memory management for all batches
        self.species = None
        self.number_of_atoms = None
        self.norm_power = None

        # TODO _n_batches is used instead of n_batches because the memory management is not yet implemented correctly
        self.n_minibatches = None  # memory management for triples generation per batch.

        self.analysis_name = "Angular_Distribution_Function"
        self.database_group = "Angular_Distribution_Function"
        self.x_label = r'$$\text{Angle} / \theta $$'
        self.y_label = r'$$\text{ADF} / a.u.$$'

    @call
    def __call__(self,
                 batch_size: int = 1,
                 n_minibatches: int = 50,
                 n_confs: int = 5,
                 r_cut: int = 6.0,
                 start: int = 1,
                 stop: int = None,
                 bins: int = 500,
                 species: list = None,
                 use_tf_function: bool = False,
                 molecules: bool = False,
                 gpu: bool = False,
                 plot: bool = True,
                 norm_power: int = 4,
                 **kwargs):
        """
        Parameters
        ----------
        batch_size : int
            Number of batches, to split the configurations into.
        n_minibatches: int
            Number of minibatches for computing the angles to split each batch into
        n_confs: int
            Number of configurations to analyse
        r_cut: float
            cutoff radius for the ADF
        start: int
            Index of the first configuration
        stop: int
            Index of the last configuration
        bins: int
            bins for the ADF
        use_tf_function: bool, default False
            activate the tf.function decorator for the minibatches. Can speed
            up the calculation significantly, but may lead to excessive use of
            memory! During the first batch, this function will be traced.
            Tracing is slow, so this might only be useful for a larger number
            of batches.
        species : list
            A list of species to use.
        norm_power: int
            The power of the normalization factor applied to the ADF histogram.
            If set to zero no distance normalization will be applied.
        molecules : bool
                if true, perform the anlaysis on molecules.
        gpu : bool
                if true, scale the memory requirements to that of the biggest
                GPU on the machine.
        plot : bool
                If true, plot the result of the analysis.

        Notes
        -----
        # TODO _n_batches is used instead of n_batches because the memory
        management is not yet implemented correctly

        """
        # Parse the parent class arguments.
        self.update_user_args(data_range=1, gpu=gpu, plot=plot)

        # Parse the user arguments.
        self.use_tf_function = use_tf_function
        self.r_cut = r_cut
        self.start = start
        self.stop = stop
        self.molecules = molecules
        self.n_confs = n_confs
        self.bins = bins
        self._batch_size = batch_size  # memory management for all batches
        self.n_minibatches = n_minibatches  # memory management for triples generation per batch.
        self.species = species
        self._check_inputs()
        self.bin_range = [0.0, 3.15]  # from 0 to a chemists pi
        self.norm_power = norm_power

    def _check_inputs(self):
        """
        Check the inputs and set defaults if necessary.

        Returns
        -------
        Updates the class attributes.
        """
        if self.stop is None:
            self.stop = self.experiment.number_of_configurations - 1

        if self.species is None:
            self.species = list(self.experiment.species)
            self._compute_number_of_atoms(self.experiment.species)

        else:
            self._compute_number_of_atoms(self.experiment.species)

        if self.molecules:
            self.species = list(self.experiment.molecules)
            self._compute_number_of_atoms(self.experiment.molecules)

    def _compute_number_of_atoms(self, reference: dict):
        """
        Compute the number of atoms total in the selected set of species.

        Parameters
        ----------
        reference : dict
                Reference dictionary in which to look for indices lists

        Returns
        -------
        Updates the number of atoms attribute.
        """
        number_of_atoms = 0
        for item in self.species:
            number_of_atoms += len(reference[item]['indices'])

        self.number_of_atoms = number_of_atoms

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
        data = self.experiment.load_matrix("Positions", path=path_list,
                                           select_slice=np.s_[:, indices])
        if len(self.species) == 1:
            return data
        else:
            return tf.concat(data, axis=0)

    def _prepare_data_structure(self):
        """
        Prepare variables and dicts for the analysis.
        Returns
        -------

        """
        sample_configs = np.linspace(self.start, self.stop, self.n_confs, dtype=np.int)

        species_indices = []
        start_index = 0
        stop_index = 0
        for species in self.species:
            stop_index += len(self.experiment.species.get(species).get('indices'))
            species_indices.append((species, start_index, stop_index))
            start_index = stop_index

        return sample_configs, species_indices

    def _prepare_generators(self, sample_configs):
        """
        Prepare the generators and compiled functions for the analysis.

        Returns
        -------

        """

        def generator():
            for idx in sample_configs:
                yield self._load_positions(idx)

        return generator

    def _prepare_triples_generator(self):
        """
        Prepare the triples generators including tf.function
        Returns
        -------

        """
        if self.use_tf_function:
            @tf.function
            def _get_triplets(x):
                return get_triplets(x,
                                    r_cut=self.r_cut,
                                    n_atoms=self.number_of_atoms,
                                    n_batches=self.n_minibatches,
                                    disable_tqdm=True)
        else:
            def _get_triplets(x):
                return get_triplets(x,
                                    r_cut=self.r_cut,
                                    n_atoms=self.number_of_atoms,
                                    n_batches=self.n_minibatches,
                                    disable_tqdm=True)

        return _get_triplets

    def _compute_rijk_matrices(self, tmp: tf.Tensor, timesteps: int):
        """
        Compute the rij matrix.

        Returns
        -------

        """
        _get_triplets = self._prepare_triples_generator()

        r_ij_flat = next(get_neighbour_list(tmp, cell=self.experiment.box_array,
                                            batch_size=1))
        r_ij_indices = get_triu_indicies(self.number_of_atoms)

        # Shape is now (n_atoms, n_atoms, 3, n_timesteps)
        r_ij_mat = tf.scatter_nd(indices=tf.transpose(r_ij_indices),
                                 updates=tf.transpose(r_ij_flat, (1, 2, 0)),
                                 shape=(self.number_of_atoms,
                                        self.number_of_atoms, 3, timesteps))

        r_ij_mat -= tf.transpose(r_ij_mat, (1, 0, 2, 3))
        r_ij_mat = tf.transpose(r_ij_mat, (3, 0, 1, 2))

        r_ijk_indices = _get_triplets(r_ij_mat)

        return r_ij_mat, r_ijk_indices

    def _compute_angles(self, species, r_ijk_indices):
        """
        Compute the angles between indices in triangle.

        Parameters
        ----------
        species
        r_ijk_indices

        Returns
        -------
        condition
        name
        """
        (i_name, i_min, i_max), (j_name, j_min, j_max), (k_name, k_min, k_max) = species
        name = f"{i_name}-{j_name}-{k_name}"

        i_condition = tf.logical_and(r_ijk_indices[:, 1] >= i_min,
                                     r_ijk_indices[:, 1] < i_max)
        j_condition = tf.logical_and(r_ijk_indices[:, 2] >= j_min,
                                     r_ijk_indices[:, 2] < j_max)
        k_condition = tf.logical_and(r_ijk_indices[:, 3] >= k_min,
                                     r_ijk_indices[:, 3] < k_max)

        condition = tf.math.logical_and(x=tf.math.logical_and(x=i_condition, y=j_condition),
                                        y=k_condition)

        return condition, name

    def _build_histograms(self, sample_configs, species_indices):
        """
        Build the adf histograms.

        Returns
        -------
        angles : dict
                A dictionary of the triples references and their histogram values.
        """
        angles = {}

        # Prepare the generators and trplets functions -- handle tf.function calls.
        generator = self._prepare_generators(sample_configs)

        # Prepare the dataset generators.
        dataset = tf.data.Dataset.from_generator(generator, output_signature=(tf.TensorSpec(shape=(None, None),
                                                                                            dtype=tf.float32)))
        dataset = dataset.batch(self._batch_size).prefetch(tf.data.AUTOTUNE)

        log.debug(f'batch_size: {self._batch_size}')

        for positions in tqdm(dataset,
                              total=self.n_confs,
                              ncols=70,
                              desc="Building histograms"):
            timesteps, atoms, _ = tf.shape(positions)
            tmp = tf.concat(positions, axis=0)
            r_ij_mat, r_ijk_indices = self._compute_rijk_matrices(tmp, timesteps)

            for species in itertools.combinations_with_replacement(species_indices, 3):
                # Select the part of the r_ijk indices, where the selected species triple occurs.
                condition, name = self._compute_angles(species, r_ijk_indices)
                tmp = tf.gather_nd(r_ijk_indices, tf.where(condition))

                # Get the indices required.
                angle_vals, pre_factor = get_angles(r_ij_mat, tmp)
                pre_factor = 1 / pre_factor ** self.norm_power
                histogram, _ = np.histogram(angle_vals,
                                            bins=self.bins,
                                            range=self.bin_range,
                                            weights=pre_factor,
                                            density=True)
                histogram = tf.cast(histogram, dtype=tf.float32)
                if angles.get(name) is not None:
                    angles.update({name: angles.get(name) + histogram})
                else:
                    angles.update({name: histogram})

        return angles

    def _compute_adfs(self, angles, species_indices):
        """
        Compute the ADF and store it.

        Parameters
        ----------
        angles : dict
                Dict of angle combinations and their histograms.
        species_indices : list
                A list of indices associated with species under consideration.

        Returns
        -------
        Updates the class, the SQL database, and plots values if required.
        """
        for species in itertools.combinations_with_replacement(species_indices, 3):
            name = f"{species[0][0]}-{species[1][0]}-{species[2][0]}"
            hist = angles.get(name)

            bin_range_to_angles = np.linspace(self.bin_range[0] * (180 / 3.14159),
                                              self.bin_range[1] * (180 / 3.14159),
                                              self.bins)

            self.data_range = self.n_confs
            log.debug(f"species are {species}")

            properties = Parameters(
                Property=self.database_group,
                Analysis=self.analysis_name,
                data_range=self.data_range,
                data=[{'angle': x, 'adf': y} for x, y in zip(bin_range_to_angles, hist)],
                Subject=[_species[0] for _species in species]
            )
            self.update_database(properties)

            if self.plot:
                self.run_visualization(
                    x_data=bin_range_to_angles,
                    y_data=hist,
                    title=f"{name} - Max: {bin_range_to_angles[tf.math.argmax(hist)]:.3f} degrees ",
                )

    def run_experimental_analysis(self):
        """
        Perform the ADF analysis.

        Notes
        -----
        # TODO select when to enable tqdm of get_triplets
        # TODO batch_size!
        # TODO allow for delete_duplicates with more then 2 species!
        """
        sample_configs, species_indices = self._prepare_data_structure()
        angles = self._build_histograms(sample_configs, species_indices)
        self._compute_adfs(angles, species_indices)
