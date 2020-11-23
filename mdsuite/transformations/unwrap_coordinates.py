"""
Atomic transformation to unwrap the simulation coordinates
"""

import numpy as np

import tensorflow as tf
import h5py as hf

class CoordinateUnwrapper:
    """ Class to unwrap particle coordinates

    attributes:
        system (object) -- The parent class from which data will be read and in which it will be saved
    """

    def __init__(self, obj, species, center_box):
        """ Standrd constuctor """

        self.system = obj  # assign the system attribute
        self.storage_path = self.system.storage_path  # get the storage path of the database
        self.analysis_name = self.system.analysis_name  # get the analysis name

        self.box_array = self.system.box_array  # re-assign the box array for cleaner code
        self.species = species  # re-assign species
        if species is None:
            self.species = self.system.species
        self.center_box = center_box  # Check if the box needs to be centered

        self.data = None  # data to be unwrapped
        self.mask = None  # image number mask

    def _center_box(self):
        """ Center the atoms in the box """

        # modify the positions in place
        self.data[:, :, 0] -= (self.box_array[0] / 2)
        self.data[:, :, 1] -= (self.box_array[1] / 2)
        self.data[:, :, 2] -= (self.box_array[2] / 2)

    def _load_data(self, species):
        """ Load the data to be unwrapped """

        self.data = np.array(self.system.load_matrix("Positions", [species]), dtype=float)

    def _calculate_difference_tensor(self):
        """ Calculate the amount particles move in each time step """

        return self.data[:, 1:] - self.data[:, :-1]

    def _build_image_mask(self):
        """ Construct a mask of image numbers """

        distance_tensor = self._calculate_difference_tensor()
        self.mask = tf.cast(tf.cast(tf.greater_equal(abs(distance_tensor),
                                        np.array(self.box_array)/2),
                                        dtype=tf.int16), dtype=tf.float64)
        self.mask = tf.multiply(tf.sign(distance_tensor), self.mask)
        self.mask = tf.map_fn(lambda x: tf.concat(([[0, 0, 0]], x), axis=0), tf.math.cumsum(self.mask, axis=1))

    def _apply_mask(self):
        """ apply the image mask to the trajectory for the unwrapping"""

        scaled_mask = self.mask*tf.convert_to_tensor(self.box_array, dtype=tf.float64)  # get the scaled mask

        self.data = self.data - scaled_mask  # apply the scaling

    def _save_unwrapped_coordinates(self, database_object, species):
        """ Save the unwrapped coordinates """

        database_object[species].create_group("Unwrapped_Positions")
        dimensions = ['x', 'y', 'z']  # labels for the database entry

        # A useful comment
        for i in range(3):
            database_object[species]["Unwrapped_Positions"].create_dataset(dimensions[i],
                                                                           data=np.array(self.data[:, :, i]))

    def unwrap_particles(self):
        """ Collect the methods in the class and unwrap the coordinates """

        with hf.File(f"{self.storage_path}/{self.analysis_name}/{self.analysis_name}.hdf5", "r+") as database:
            for species in self.system.species:
                if "Unwrapped_Positions" in database[species].keys():
                    print(f"Unwrapped positions exists for {species}, using the saved coordinates")
                else:
                    self._load_data(species)  # load the data

                    if self.center_box:
                        self._center_box()  # center the data if required

                    self._build_image_mask()  # build the image mask
                    self._apply_mask()  # Apply the mask and unwrap the coordinates
                    self._save_unwrapped_coordinates(database, species)  # save the newly unwrapped coordinates