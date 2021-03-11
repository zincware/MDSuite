import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil

import mdsuite as mds # Cool name for professionalism purposes

new_case = False

if new_case:
    try:
        shutil.rmtree('Argon')
    except FileNotFoundError:
        pass



argon = mds.Experiment(analysis_name="Argon", time_step=2, temperature=94.4, units='real')

if new_case:
    argon.add_data(trajectory_file='../gk_data.lmp_traj')

# argon.help_computations_args('EinsteinDiffusionCoefficients') # auxiliary function to help on the extra arguments
# argon.run_computation('EinsteinDiffusionCoefficients', species=list(argon.species.keys()),data_range=50, plot=False, singular=True, distinct=False)
argon.run_computation('GreenKuboThermalConductivity', data_range=2500, plot=True, correlation_time=2)
argon.run_computation('GreenKuboViscosity', data_range=2500, plot=True, correlation_time=2)

# print(argon.results)
# argon.dump_results_json()