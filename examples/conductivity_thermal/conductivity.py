from mdsuite.experiment.conductivity_thermal import ProjectThermal

argon = ProjectThermal(analysis_name="Argon_70",
                       new_project=True,
                       storage_path=".",
                       temperature=70.0,
                       time_step=4,
                       time_unit=1e-15,
                       filename="../trajectory_files/flux.dat",
                       length_unit=1e-10)

argon._save_class()
argon.print_class_attributes()

argon.Green_Kubo_Conductivity_Thermal(800, plot=True)
argon.print_class_attributes()

