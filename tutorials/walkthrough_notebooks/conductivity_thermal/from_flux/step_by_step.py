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

    # the rename cols is only used to show here how it works, but in this case, the lammps file has the same column names...
    argon.add_data(trajectory_file='../flux_1.lmp_flux', file_format='lammps_flux', rename_cols={'Thermal_Flux':['c_flux_thermal[1]', 'c_flux_thermal[2]', 'c_flux_thermal[3]']})
    argon.run_computation('GreenKuboThermalConductivity', data_range=2000, plot=True, correlation_time=3)
