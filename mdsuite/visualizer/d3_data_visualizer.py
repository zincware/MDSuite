"""
Visualizer for three dimensional data.
"""
import open3d as o3d
import open3d.visualization.gui as gui
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

        self._build_app()

    def _build_app(self):
        """
        Build the app window and update the class.

        Returns
        -------
        Updates the class.
        """
        self.app = gui.Application.instance
        self.app.initialize()

        self.vis = o3d.visualization.O3DVisualizer("MDSuite Visualizer",
                                                   1024,
                                                   768)
        self.vis.show_settings = True
        # Add meshes
        self.vis.reset_camera_to_default()

        self.app.add_window(self.vis)

    def plot(self):
        """
        Plot the data.

        Returns
        -------

        """
        self.vis.draw_geometries([self.point_cloud])
