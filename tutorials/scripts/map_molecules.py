"""
Example file for how to map atoms from a simulation to defined molecules.
"""

import sys
sys.path.insert(1, "../../")

import mdsuite as mds
from mdsuite.transformations.map_molecules import MolecularMap
import shutil

def main():
    """
    Analyze the trajectory.
    """
    try:
       shutil.rmtree('emim')
    except:
       pass
    filename = "../data/trajectory_files/emim_f.dump"
    emim = mds.Experiment(analysis_name='emim', time_step=1.0, units='real', temperature=303.15)
    emim.add_data(trajectory_file=filename)
    emim.species['N']['mass'] = 14.007
    emim.species['C']['mass'] = 12.011
    emim.species['H']['mass'] = 1.008
    emim.species['F']['mass'] = 18.998
    #emim.run_computation.EinsteinHelfandIonicConductivity(plot=True, data_range=50)
    #emim.run_computation.GreenKuboIonicConductivity(plot=True, data_range=50)
    emim.perform_transformation("UnwrapCoordinates")
    mapper = MolecularMap(emim, molecules={'emim': {'smiles': 'CCN1C=C[N+](+C1)C', 'amount': 20}})
    mapper.run_transformation()
    emim.perform_transformation('WrapCoordinates', species=['emim'])
    print(emim.molecules)
    #emim.write_xyz(species=['emim'], name='emim', dump_property='Positions')
    #emim.write_xyz(species=['C', 'H', 'N'], name='atomistic', dump_property='Positions')


if __name__ == "__main__":
    main()
