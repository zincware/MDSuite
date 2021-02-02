"""
Atomic transformation to unwrap the simulation coordinates.

Summary
-------
When performing analysis on the dynamics of a system, it often becomes necessary to reverse the effects of periodic
boundary conditions and track atoms across the box edges. This method uses the box-jump algorithm, whereing particle
positions jumps of more than a half of the box are counted as a crossing of the boundary, to allow the particles to
propagate on into space.
"""

import numpy as np
import os

import tensorflow as tf
import h5py as hf

from mdsuite.transformations.transformations import Transformations
from mdsuite.database.database import Database


class CoordinateUnwrapper(Transformations):
    """
    Class to unwrap particle coordinates

    Attributes
    ----------
    system : object
            The parent class from which data will be read and in which it will be saved.

    species : list
            species of atoms to unwrap.

    center_box : bool
            Decision whether or not to center the positions in the box before performing the unwrapping. The default
            value is set to True as this is most common in simulations.

    storage_path : str
            Path to the data in the database, taken directly from the system attribute.

    analysis_name : str
            Name of the analysis, taken directly from the system attribute.

    box_array : list
            Box array of the simulation.

    data : tf.tensor
            Data being unwrapped.

    mask : tf.tensor
            Mask to select and transform crossed data.
    """

    def __init__(self, obj, species, center_box=True):
        """
        Standard constructor

        Parameters
        ----------
        obj : object
                Experiment object to use and update.
        species : list
                List of species to perform unwrapping on
        center_box : bool
                If true, the box coordinates will be centered before the unwrapping occurs
        """

        super().__init__()
        self.system = obj  # assign the system attribute
        self.storage_path = self.system.storage_path  # get the storage path of the database
        self.analysis_name = self.system.analysis_name  # get the analysis name

        self.box_array = self.system.box_array  # re-assign the box array for cleaner code
        self.species = species  # re-assign species
        if species is None:
            self.species = list(self.system.species)
        self.center_box = center_box  # Check if the box needs to be centered

        self.data = None  # data to be unwrapped
        self.mask = None  # image number mask

    def _center_box(self):
        """
        Center the atoms in the box

        Shift all of the positions by half a box length to move the origin of the box to the center.
        """

        # modify the positions in place
        self.data[:, :, 0] -= (self.box_array[0] / 2)
        self.data[:, :, 1] -= (self.box_array[1] / 2)
        self.data[:, :, 2] -= (self.box_array[2] / 2)

    def _load_data(self, species):
        """
        Load the data to be unwrapped
        """

        self.data = np.array(self.system.load_matrix("Positions", [species]), dtype=float)

    def _calculate_difference_tensor(self):
        """
        Calculate the amount particles move in each time step

        Returns
        -------
        distance tensor : tf.tensor
                Returns the tensor of distances in time.
        """

        return self.data[:, 1:] - self.data[:, :-1]

    def _build_image_mask(self):
        """
        Construct a mask of image numbers
        """

        distance_tensor = self._calculate_difference_tensor()  # get the distance tensor

        # Find all distance greater than half a box length and set them to integers.
        self.mask = tf.cast(tf.cast(tf.greater_equal(abs(distance_tensor),
                                                     np.array(self.box_array) / 2),
                                    dtype=tf.int16), dtype=tf.float64)

        self.mask = tf.multiply(tf.sign(distance_tensor), self.mask)  # get the correct image sign

        # Sum over consecutive box jumps to gather the correct number of times the particles jumped.
        self.mask = tf.map_fn(lambda x: tf.concat(([[0, 0, 0]], x), axis=0), tf.math.cumsum(self.mask, axis=1))

    def _apply_mask(self):
        """
        Apply the image mask to the trajectory for the unwrapping
        """

        scaled_mask = self.mask * tf.convert_to_tensor(self.box_array, dtype=tf.float64)  # get the scaled mask

        self.data = self.data - scaled_mask  # apply the scaling

    def _save_unwrapped_coordinates(self, database_object, species, database):
        """
        Save the unwrapped coordinates
        """

        path = os.path.join(species, 'Unwrapped_Positions')
        dataset_structure = {species: {'Unwrapped_Positions': tuple(np.shape(self.data))}}
        database.add_dataset(dataset_structure, database_object)  # add the dataset to the database as resizeable
        data_structure = {path: {'indices': np.s_[:], 'columns': [0, 1, 2], 'length': len(self.data)}}
        database.add_data(data=self.data,
                          structure=data_structure,
                          database=database_object,
                          start_index=0,
                          batch_size=np.shape(self.data)[1],
                          tensor=True)

    def unwrap_particles(self):
        """
        Collect the methods in the class and unwrap the coordinates
        """

        database = Database(name=os.path.join(self.system.database_path, "database.hdf5"), architecture='simulation')
        db_object = database.open()  # open a database object
        for species in self.species:

            exists = database.check_existence(os.path.join(species, "Unwrapped_Positions"))
            # Check if the data has already been unwrapped
            if exists:
                print(f"Unwrapped positions exists for {species}, using the saved coordinates")
            else:
                self._load_data(species)  # load the data to be unwrapped

                # Center the data if required
                if self.center_box:
                    self._center_box()

                self._build_image_mask()  # build the image mask
                self._apply_mask()  # Apply the mask and unwrap the coordinates
                self._save_unwrapped_coordinates(db_object, species, database)  # save the newly unwrapped coordinates

        database.close(db_object)
        self.system.memory_requirements = database.get_memory_information()
        self.system.save_class()  # update the class state

    def run_transformation(self):
        """
        Perform the transformation.
        """
        self.unwrap_particles()  # run the transformation
