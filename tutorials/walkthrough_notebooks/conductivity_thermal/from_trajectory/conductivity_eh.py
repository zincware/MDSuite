"""
Compute thermal conductivity.
"""
import mdsuite as mds

argon = mds.Experiment(analysis_name="Argon", time_step=1, temperature=70.0, units='real')
argon.add_data(trajectory_file='../gk_data.lmp_traj')
argon.run_computation.EinsteinHelfandThermalConductivity(data_range=4000, plot=True, correlation_time=3)
