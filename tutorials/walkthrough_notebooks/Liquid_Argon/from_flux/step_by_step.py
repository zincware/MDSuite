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
                             temperature=94.4,
                             time_step=2,
                             units='real')

    argon.add_data(trajectory_file='../flux_1.lmp_flux', file_format='lammps_flux')
    argon.run_computation('GreenKuboThermalConductivityFlux', data_range=19990, plot=True)
