"""
Compute the viscosity and flux from a trajectory from Green-Kubo
"""
import mdsuite as mds

argon = mds.Experiment(analysis_name="Argon", time_step=2, temperature=94.4, units='real')

argon.add_data(trajectory_file='../gk_data.lmp_traj')

argon.run_computation('EinsteinDiffusionCoefficients', species=list(argon.species.keys()),data_range=50, plot=False,
                      singular=True, distinct=False)
argon.run_computation('GreenKuboThermalConductivity', data_range=2500, plot=True, correlation_time=2)
argon.run_computation('GreenKuboViscosity', data_range=2500, plot=True, correlation_time=2)
