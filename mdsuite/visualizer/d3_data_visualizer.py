"""
Visualizer for three dimensional data.
"""
import open3d as o3d
import open3d.visualization.gui as gui
import tensorflow as tf


class DataVisualizer3D:
    """
    Class for the visualizer of three dimensional data.
    """

    def __init__(self,
                 data: tf.Tensor,
                 title: str,
                 center: str = None
                 ):
        """
        Constructor for the data visualizer.

        Parameters
        ----------
        data : tf.Tensor
                data to plot.
        center : str
                centre molecule to be displayed (optional)
        title : str
                title of the plot.
        """
        self.data =data
        self.title = title
        self.center = center

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

    def _add_center(self):
        """
        Add a rendering to the (0, 0, 0) point in the plot.

        Returns
        -------
        Updates the plot.
        """

        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1,
                                                       resolution=5)
        mesh.compute_vertex_normals()

    def plot(self):
        """
        Plot the data.

        Returns
        -------

        """
        #self.vis.draw_geometries([self.point_cloud])
        self.vis.add_geometry("Points", self.point_cloud)

        if self.center is not None:
            self._add_center()

        self.vis.reset_camera_to_default()
        self.app.run()
