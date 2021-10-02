"""
MDSuite: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
Module to perform the calculation of the angular distribution function (ADF). The ADF
describes the average distribution of angles between three particles of species a, b,
and c. Note that a, b, and c may all be the same species, e.g. Na-Na-Na.
"""
from abc import ABC
import logging
import tensorflow as tf
import itertools
import numpy as np
from tqdm import tqdm
from typing import Union
from mdsuite.calculators.calculator import Calculator, call
from mdsuite.utils.neighbour_list import (
    get_neighbour_list,
    get_triu_indicies,
    get_triplets,
)
from mdsuite.utils.linalg import get_angles
from mdsuite.utils.meta_functions import join_path
from mdsuite.database.scheme import Computation
from dataclasses import dataclass
from mdsuite.database import simulation_properties

log = logging.getLogger(__name__)


@dataclass
class Args:
    number_of_bins: int
    number_of_configurations: int
    correlation_time: int
    atom_selection: np.s_
    data_range: int
    cutoff: float
    start: int
    norm_power: Union[int, float]
    stop: int
    species: list
    molecules: bool


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
        activate the tf.function decorator for the minibatches. Can speed up the
        calculation significantly, but may lead to excessive use of memory! During the
        first batch, this function will be traced. Tracing is slow, so this might only
        be useful for a larger number of batches.

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
        self.scale_function = {"quadratic": {"outer_scale_factor": 10}}
        self.loaded_property = "Positions"

        self.use_tf_function = None
        self.cutoff = None
        self.start = None
        self.molecules = None
        self.experimental = True
        self.stop = None
        self.number_of_configurations = None
        self.number_of_bins = None
        self.bin_range = None
        self._batch_size = None  # memory management for all batches
        self.species = None
        self.number_of_atoms = None
        self.norm_power = None
        self.result_series_keys = ["angle", "adf"]

        # TODO _n_batches is used instead of n_batches because the memory management is
        #  not yet implemented correctly
        self.minibatch = None  # memory management for triples generation per batch.

        self.analysis_name = "Angular_Distribution_Function"
        self.database_group = "Angular_Distribution_Function"
        self.x_label = r"$$\text{Angle} / \theta $$"
        self.y_label = r"$$\text{ADF} / a.u.$$"

    @call
    def __call__(
        self,
        batch_size: int = 1,
        minibatch: int = 50,
        number_of_configurations: int = 5,
        cutoff: int = 6.0,
        start: int = 1,
        stop: int = None,
        number_of_bins: int = 500,
        species: list = None,
        use_tf_function: bool = False,
        molecules: bool = False,
        gpu: bool = False,
        plot: bool = True,
        norm_power: int = 4,
        **kwargs,
    ):
        """
        Parameters
        ----------
        batch_size : int
            Number of batches, to split the configurations into.
        minibatch: int
            Number of minibatches for computing the angles to split each batch into
        number_of_configurations: int
            Number of configurations to analyse
        cutoff: float
            cutoff radius for the ADF
        start: int
            Index of the first configuration
        stop: int
            Index of the last configuration
        number_of_bins: int
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
                if true, perform the analysis on molecules.
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
        # set args that will affect the computation result
        self.args = Args(
            number_of_bins=number_of_bins,
            cutoff=cutoff,
            start=start,
            stop=stop,
            atom_selection=np.s_[:],
            data_range=1,
            correlation_time=1,
            molecules=molecules,
            species=species,
            number_of_configurations=number_of_configurations,
            norm_power=norm_power
        )

        # Parse the user arguments.
        self.use_tf_function = use_tf_function
        self.cutoff = cutoff
        self.gpu = gpu
        self.plot = plot
        self._batch_size = batch_size  # memory management for all batches
        self.minibatch = (
            minibatch  # memory management for triples generation per batch.
        )
        self.bin_range = [0.0, 3.15]  # from 0 to a chemists pi
        self.norm_power = norm_power

    def check_inputs(self):
        """
        Check the inputs and set defaults if necessary.

        Returns
        -------
        Updates the class attributes.
        """
        if self.args.stop is None:
            self.args.stop = self.experiment.number_of_configurations - 1

        # Get the correct species out.
        if self.args.species is None:
            if self.args.molecules:
                self.args.species = list(self.experiment.molecules)
            else:
                self.args.species = list(self.experiment.species)

        else:
            self._compute_number_of_atoms(self.experiment.species)

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
            number_of_atoms += len(reference[item]["indices"])

        self.number_of_atoms = number_of_atoms

    def _prepare_data_structure(self):
        """
        Prepare variables and dicts for the analysis.
        Returns
        -------

        """
        sample_configs = np.linspace(
            self.start, self.stop, self.number_of_configurations, dtype=np.int
        )

        species_indices = []
        start_index = 0
        stop_index = 0
        for species in self.species:
            stop_index += len(self.experiment.species.get(species).get("indices"))
            species_indices.append((species, start_index, stop_index))
            start_index = stop_index

        return sample_configs, species_indices

    def _prepare_triples_generator(self):
        """
        Prepare the triples generators including tf.function
        Returns
        -------

        """
        if self.use_tf_function:

            @tf.function
            def _get_triplets(x):
                return get_triplets(
                    x,
                    r_cut=self.cutoff,
                    n_atoms=self.number_of_atoms,
                    n_batches=self.minibatch,
                    disable_tqdm=True,
                )

        else:

            def _get_triplets(x):
                return get_triplets(
                    x,
                    r_cut=self.cutoff,
                    n_atoms=self.number_of_atoms,
                    n_batches=self.minibatch,
                    disable_tqdm=True,
                )

        return _get_triplets

    def _compute_rijk_matrices(self, tmp: tf.Tensor, timesteps: int):
        """
        Compute the rij matrix.

        Returns
        -------

        """
        _get_triplets = self._prepare_triples_generator()

        r_ij_flat = next(
            get_neighbour_list(tmp, cell=self.experiment.box_array, batch_size=1)
        )
        r_ij_indices = get_triu_indicies(self.number_of_atoms)

        # Shape is now (n_atoms, n_atoms, 3, n_timesteps)
        r_ij_mat = tf.scatter_nd(
            indices=tf.transpose(r_ij_indices),
            updates=tf.transpose(r_ij_flat, (1, 2, 0)),
            shape=(self.number_of_atoms, self.number_of_atoms, 3, timesteps),
        )

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

        i_condition = tf.logical_and(
            r_ijk_indices[:, 1] >= i_min, r_ijk_indices[:, 1] < i_max
        )
        j_condition = tf.logical_and(
            r_ijk_indices[:, 2] >= j_min, r_ijk_indices[:, 2] < j_max
        )
        k_condition = tf.logical_and(
            r_ijk_indices[:, 3] >= k_min, r_ijk_indices[:, 3] < k_max
        )

        condition = tf.math.logical_and(
            x=tf.math.logical_and(x=i_condition, y=j_condition), y=k_condition
        )

        return condition, name

    def _build_histograms(self, positions, species_indices, angles):
        """
        Build the adf histograms.

        Returns
        -------
        angles : dict
                A dictionary of the triples references and their histogram values.
        """
        timesteps, atoms, _ = tf.shape(positions)
        tmp = tf.concat(positions, axis=0)
        r_ij_mat, r_ijk_indices = self._compute_rijk_matrices(tmp, timesteps)

        for species in itertools.combinations_with_replacement(species_indices, 3):
            # Select the part of the r_ijk indices, where the selected species
            # triple occurs.
            condition, name = self._compute_angles(species, r_ijk_indices)
            tmp = tf.gather_nd(r_ijk_indices, tf.where(condition))

            # Get the indices required.
            angle_vals, pre_factor = get_angles(r_ij_mat, tmp)
            pre_factor = 1 / pre_factor ** self.norm_power
            histogram, _ = np.histogram(
                angle_vals,
                bins=self.number_of_bins,
                range=self.bin_range,
                weights=pre_factor,
                density=True,
            )
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

            bin_range_to_angles = np.linspace(
                self.bin_range[0] * (180 / 3.14159),
                self.bin_range[1] * (180 / 3.14159),
                self.number_of_bins,
            )

            self.selected_species = [_species[0] for _species in species]

            self.data_range = self.number_of_configurations
            log.debug(f"species are {species}")

            data = {
                self.result_series_keys[0]: bin_range_to_angles.tolist(),
                self.result_series_keys[1]: hist.numpy().tolist(),
            }

            self.queue_data(data=data, subjects=self.selected_species)

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

    def plot_data(self, data):
        """Plot data"""
        for selected_species, val in data.items():
            bin_range_to_angles = np.linspace(
                self.bin_range[0] * (180 / 3.14159),
                self.bin_range[1] * (180 / 3.14159),
                len(val[self.result_series_keys[0]]),
            )
            title_value = bin_range_to_angles[
                tf.math.argmax(val[self.result_series_keys[1]])
            ]

            self.run_visualization(
                x_data=np.array(val[self.result_series_keys[0]]),
                y_data=np.array(val[self.result_series_keys[1]]),
                title=(f"{selected_species} - Max:" f" {title_value:.3f} degrees "),
            )

    def _format_data(self, batch: tf.Tensor, keys: list) -> tf.Tensor:
        """
        Format the loaded data for use in the rdf calculator.

        The RDF requires a reshaped dataset. The generator will load a default
        dict oriented type. This method restructures the data to be used in the
        calculator.

        Parameters
        ----------
        batch : tf.Tensor
                A batch of data to transform.
        keys : list
                Dict keys to extract from the data.

        Returns
        -------

        """
        formatted_data = []
        for item in keys:
            formatted_data.append(batch[item])

        if len(self.args.species) == 1:
            return tf.cast(formatted_data[0], dtype=self.dtype)
        else:
            return tf.cast(tf.concat(formatted_data, axis=0), dtype=self.dtype)

    def run_calculator(self):
        """
        Run the analysis.

        Returns
        -------

        """

        dict_keys, split_arr, batch_tqm = self.prepare_computation()

        # Get the batch dataset
        batch_ds = self.get_batch_dataset(
            subject_list=self.args.species, loop_array=split_arr, correct=True
        )

        angles = {}

        # Loop over the batches.
        for idx, batch in tqdm(enumerate(batch_ds), ncols=70, disable=batch_tqm):
            angles = self._build_histograms(positions=batch, angles=angles)



