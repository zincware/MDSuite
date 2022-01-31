import mdsuite as mds

from mdsuite.calculators.structure_factor import StructureFactor


project = mds.Project("struc_fac")
project.add_experiment(
    name="NaCl_1399K",
    timestep=2.0,
    temperature=1400.0,
    units="metal",
    simulation_data="short.lammpstraj",
)

"""Make sure to give your particles the right charge,
otherwise the calculated structure factor will be incorrect"""
project.experiments["NaCl_1399K"].set_charge("Na", [+1])
project.experiments["NaCl_1399K"].set_charge("Cl", [-1])

# You also need to use the correct mass, but this is usually already set
# project.experiments['NaCl_1399K'].set_mass('Cl',[50])


project.run.StructureFactor(plot=True)
