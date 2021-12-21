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
import logging

import numpy as np
import tensorflow as tf

from mdsuite.transformations.transformations import Transformations
from mdsuite.utils.meta_functions import join_path

log = logging.getLogger(__name__)


class CoordinateWrapper(Transformations):
    """
    Class to unwrap particle coordinates

    Attributes
    ----------
    experiment : object
            The experiment class from which tensor_values will be read and in
            which it will be saved.

    species : list
            species of atoms to unwrap.

    center_box : bool
            Decision whether or not to center the positions in the box before
            performing the unwrapping. The default value is set to True as
            this is most common in simulations.

    storage_path : str
            Path to the tensor_values in the database_path, taken directly
            from the system attribute.

    analysis_name : str
            Name of the analysis, taken directly from the system attribute.

    box_array : list
            Box array of the simulation.

    data : tf.tensor
            Data being unwrapped.

    mask : tf.tensor
            Mask to select and transform crossed tensor_values.
    scale_function : dict
            A dictionary referencing the memory/time scaling function
            of the transformation.
    """

    def __init__(self, species: list = None, center_box: bool = True):
        """
        Standard constructor

        Parameters
        ----------
        species : list
                List of species to perform unwrapping on
        center_box : bool
                If true, the box coordinates will be centered before the
                unwrapping occurs
        """
        super().__init__()

        self.scale_function = {"linear": {"scale_factor": 5}}
        self.center_box = center_box
        self.species = species  # re-assign species

    def update_from_experiment(self):
        """
        Update information from the experiment.

        Returns
        -------
        Updates the class state.
        """
        self.storage_path = self.experiment.storage_path
        self.analysis_name = self.experiment.name

        self.box_array = self.experiment.box_array
        if self.species is None:
            self.species = list(self.experiment.species)

        self.data = None
        self.mask = None

    def _load_data(self, species):
        """
        Load the tensor_values to be unwrapped
        """
        path_list = [join_path(species, "Unwrapped_Positions")]
        self.data = self.database.load_data(
            path_list=path_list, select_slice=np.s_[:], dictionary=True
        )[path_list[0]]

    def _center_box(self):
        """
        Center the box to match other coordinates.
        Returns
        -------
        adjusts the self.data attribute
        """
        self.data = np.array(self.data)
        self.data[:, :, 0] += self.box_array[0] / 2
        self.data[:, :, 1] += self.box_array[1] / 2
        self.data[:, :, 2] += self.box_array[2] / 2

    def _build_image_mask(self):
        """
        Construct a mask of image numbers.

        This works by finding all distances greater than half the box length
        and set them to a float in order to build the mask and set the values
        to the correct sign.
        """
        self.mask = tf.cast(
            tf.cast(
                tf.greater_equal(abs(self.data), np.array(self.box_array) / 2),
                dtype=tf.int16,
            ),
            dtype=tf.float64,
        )
        self.mask = tf.math.floordiv(x=abs(self.data), y=np.array(self.box_array) / 2)

        self.mask = tf.multiply(
            tf.sign(self.data), self.mask
        )  # get the correct image sign

    def _apply_mask(self):
        """
        Apply the image mask to the trajectory for the wrapping
        """

        scaled_mask = self.mask * tf.convert_to_tensor(self.box_array, dtype=tf.float64)

        self.data = self.data - scaled_mask  # apply the scaling

    def wrap_particles(self):
        """
        Collect the methods in the class and unwrap the coordinates
        """

        for species in self.species:

            exists = self.database.check_existence(join_path(species, "Positions"))
            # Check if the tensor_values has already been unwrapped
            if exists:
                log.info(
                    f"Wrapped positions exists for {species}, using the saved coordinates"
                )
            else:
                self._load_data(species)
                self.data = tf.convert_to_tensor(self.data)
                self._build_image_mask()  # build the image mask
                self._apply_mask()  # Apply the mask and unwrap the coordinates
                if self.center_box:
                    self._center_box()
                path = join_path(species, "Positions")
                dataset_structure = {species: {"Positions": tuple(np.shape(self.data))}}
                self.database.add_dataset(dataset_structure)
                data_structure = {
                    path: {
                        "indices": np.s_[:],
                        "columns": [0, 1, 2],
                        "length": len(self.data),
                    }
                }
                self._save_output(
                    data=self.data,
                    data_structure=data_structure,
                    index=0,
                    batch_size=np.shape(self.data)[1],
                    system_tensor=False,
                    tensor=True,
                )

        self.experiment.memory_requirements = self.database.get_memory_information()

    def run_transformation(self):
        """
        Perform the transformation.
        """
        self.wrap_particles()
