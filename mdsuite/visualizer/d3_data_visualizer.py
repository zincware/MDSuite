"""
Visualizer for three dimensional data.
"""
import open3d as o3d
import numpy as np


class DataVisualizer3D:
    """
    Class for the visualizer of three dimensional data.
    """

    def __init__(self,
                 data,
                 title: str):
        """
        Constructor for the data visualizer.

        Parameters
        ----------
        data : np.ndarray
                data to plot.
        title : str
                title of the plot.
        """
        self.data =data
        self.title = title

        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(self.data)

    def plot(self):
        """
        Plot the data.

        Returns
        -------

        """
        o3d.visualization.draw_geometries([self.point_cloud])
