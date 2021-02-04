import shutil

from mdsuite.experiment.experiment import Experiment

if __name__ == '__main__':

    try:
        new_case = True
        shutil.rmtree('Argon_70')
    except FileNotFoundError:
        pass
    argon = Experiment(analysis_name="Argon_70",
                             storage_path=".",
                             temperature=70.0,
                             time_step=4,
                             units='real')

    argon.add_data(trajectory_file='../flux_1.lmp_flux', file_format='lammps_flux', rename_cols={'Flux_Thermal':['c_flux[1]', 'c_flux[2]', 'c_flux[3]']})
    argon.run_computation('GreenKuboThermalConductivityFlux', data_range=8000, plot=False)
