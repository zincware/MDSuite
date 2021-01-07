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

import tensorflow as tf
import h5py as hf


class CoordinateUnwrapper:
    """
    Class to unwrap particle coordinates

    Parameters
    ----------
    obj, system : object
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

    def __init__(self, obj, species, center_box):
        """
        Standard constructor
        """

        self.system = obj                               # assign the system attribute
        self.storage_path = self.system.storage_path    # get the storage path of the database
        self.analysis_name = self.system.analysis_name  # get the analysis name

        self.box_array = self.system.box_array          # re-assign the box array for cleaner code
        self.species = species                          # re-assign species
        if species is None:
            self.species = self.system.species
        self.center_box = center_box                    # Check if the box needs to be centered

        self.data = None                                # data to be unwrapped
        self.mask = None                                # image number mask

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
                                        np.array(self.box_array)/2),
                                        dtype=tf.int16), dtype=tf.float64)

        self.mask = tf.multiply(tf.sign(distance_tensor), self.mask)  # get the correct image sign

        # Sum over consecutive box jumps to gather the correct number of times the particles jumped.
        self.mask = tf.map_fn(lambda x: tf.concat(([[0, 0, 0]], x), axis=0), tf.math.cumsum(self.mask, axis=1))

    def _apply_mask(self):
        """
        Apply the image mask to the trajectory for the unwrapping
        """

        scaled_mask = self.mask*tf.convert_to_tensor(self.box_array, dtype=tf.float64)  # get the scaled mask

        self.data = self.data - scaled_mask  # apply the scaling

    def _save_unwrapped_coordinates(self, database_object, species):
        """
        Save the unwrapped coordinates
        """

        database_object[species].create_group("Unwrapped_Positions")  # Create a new group
        dimensions = ['x', 'y', 'z']                                  # labels for the database entry

        # Store the data in the x, y, and z database groups.
        for i in range(3):
            database_object[species]["Unwrapped_Positions"].create_dataset(dimensions[i],
                                                                           data=np.array(self.data[:, :, i]))

    def unwrap_particles(self):
        """
        Collect the methods in the class and unwrap the coordinates
        """

        with hf.File(f"{self.storage_path}/{self.analysis_name}/{self.analysis_name}.hdf5", "r+") as database:
            for species in self.system.species:

                # Check if the data has already been unwrapped
                if "Unwrapped_Positions" in database[species].keys():
                    print(f"Unwrapped positions exists for {species}, using the saved coordinates")
                else:
                    self._load_data(species)  # load the data to be unwrapped

                    # Center the data if required
                    if self.center_box:
                        self._center_box()

                    self._build_image_mask()                             # build the image mask
                    self._apply_mask()                                   # Apply the mask and unwrap the coordinates
                    self._save_unwrapped_coordinates(database, species)  # save the newly unwrapped coordinates
