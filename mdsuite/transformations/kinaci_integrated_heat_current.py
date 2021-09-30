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
"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from typing import Tuple
from mdsuite.transformations.transformations import Transformations
from mdsuite.utils.meta_functions import join_path


class KinaciIntegratedHeatCurrent(Transformations):
    """
    Class to generate and store the ionic current of a experiment

    Attributes
    ----------
    experiment : object
            Experiment this transformation is attached to.
    scale_function : dict
            A dictionary referencing the memory/time scaling function of the
            transformation.
    """

    def __init__(self, experiment: object):
        """
        Constructor for the Ionic current calculator.

        Parameters
        ----------
        experiment : object
                Experiment this transformation is attached to.
        """
        super().__init__(experiment)
        self.scale_function = {"linear": {"scale_factor": 5}}

    def _transformation(
        self, data: tf.Tensor, cumul_integral, batch_size
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Calculate the integrated thermal current of the system.

        Returns
        -------
        Integrated heat current : tf.Tensor
                The values for the integrated heat current.
        """
        integral = tf.tile(cumul_integral, (1, batch_size))
        system_current = tf.zeros((batch_size, 3), dtype=tf.float64)

        for species in self.experiment.species:
            positions_path = str.encode(join_path(species, "Unwrapped_Positions"))
            velocity_path = str.encode(join_path(species, "Velocities"))
            force_path = str.encode(join_path(species, "Forces"))
            pe_path = str.encode(join_path(species, "PE"))

            integrand = tf.einsum("ijk,ijk->ij", data[force_path], data[velocity_path])
            # add here the value from the previous iteration to all the steps in
            # this batch.
            integral += (
                tf.cumsum(integrand, axis=1)
                * self.experiment.time_step
                * self.experiment.sample_rate
            )

            r_k = tf.einsum("ijk,ij->jk", data[positions_path], integral)
            r_p = tf.einsum("ijk,ijm->jm", data[pe_path], data[positions_path])

            system_current += r_k + r_p

        cumul_integral = tf.expand_dims(integral[:, -1], axis=1)
        return system_current, cumul_integral

    def _prepare_database_entry(self) -> dict:
        """
        Add the relevant tensor_values sets and groups in the database_path

        Returns
        -------
        data_structure:
                tensor_values structure for use in saving the tensor_values to the
                database_path.
        """

        number_of_configurations = self.experiment.number_of_configurations
        path = join_path("Kinaci_Heat_Current", "Kinaci_Heat_Current")
        dataset_structure = {path: (number_of_configurations, 3)}
        self.database.add_dataset(
            dataset_structure
        )  # add a new dataset to the database_path
        data_structure = {path: {"indices": np.s_[:], "columns": [0, 1, 2]}}

        return data_structure

    def _compute_thermal_flux(self):
        """
        Loop over batches and compute the dipole moment
        Returns
        -------
        Updates the simulation database
        """
        data_structure = self._prepare_database_entry()
        type_spec = {}

        positions_path = [
            join_path(species, "Unwrapped_Positions")
            for species in self.experiment.species
        ]
        velocities_path = [
            join_path(species, "Velocities") for species in self.experiment.species
        ]
        forces_path = [
            join_path(species, "Forces") for species in self.experiment.species
        ]
        pe_path = [join_path(species, "PE") for species in self.experiment.species]
        data_path = np.concatenate(
            (positions_path, velocities_path, forces_path, pe_path)
        )

        self._prepare_monitors(data_path)
        # update the dictionary (mutable object)
        type_spec = self._update_species_type_dict(type_spec, positions_path, 3)
        type_spec = self._update_species_type_dict(type_spec, velocities_path, 3)
        type_spec = self._update_species_type_dict(type_spec, forces_path, 3)
        type_spec = self._update_species_type_dict(type_spec, pe_path, 1)
        type_spec[str.encode("data_size")] = tf.TensorSpec(None, dtype=tf.int16)
        batch_generator, batch_generator_args = self.data_manager.batch_generator(
            dictionary=True, remainder=True
        )
        data_set = tf.data.Dataset.from_generator(
            batch_generator, args=batch_generator_args, output_signature=type_spec
        )

        data_set = data_set.prefetch(tf.data.experimental.AUTOTUNE)

        cumul_integral = tf.zeros(
            [self.experiment.number_of_atoms, 1], dtype=tf.float64
        )

        # x is batch of data.
        for idx, x in tqdm(
            enumerate(data_set),
            ncols=70,
            desc="Kinaci Integrated Current",
            total=self.n_batches,
        ):
            current_batch_size = int(x[str.encode("data_size")])
            data, cumul_integral = self._transformation(
                x, cumul_integral=cumul_integral, batch_size=current_batch_size
            )
            self._save_coordinates(
                data, idx * self.batch_size, current_batch_size, data_structure
            )

    def run_transformation(self):
        """
        Run the ionic current transformation
        Returns
        -------
        Nothing.
        """
        self._compute_thermal_flux()  # run the transformation.
        self.experiment.memory_requirements = self.database.get_memory_information()
