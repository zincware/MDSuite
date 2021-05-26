"""
Compute viscosity from Green-Kubo
"""
import mdsuite as mds # Cool name for professionalism purposes

argon = mds.Experiment(analysis_name="Argon", time_step=2, temperature=94.4, units='real')
argon.add_data(trajectory_file='../gk_data.lmp_traj')
argon.run_computation('GreenKuboViscosity', data_range=19980, plot=True)
