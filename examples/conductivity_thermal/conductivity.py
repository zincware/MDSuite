import sys
from mdsuite.conductivity_thermal import ProjectThermal

argon = ProjectThermal(analysis_name="Argon_70",
                       new_project=True,
                       storage_path=".",
                       temperature=70.0,
                       time_step=4,
                       time_unit=1e-15,
                       filename="flux.dat",
                       length_unit=1e-10)

argon.Save_Class()
argon.Print_Class_Attributes()

argon.Green_Kubo_Conductivity_Thermal(800, plot=True)
argon.Print_Class_Attributes()

