import mdsuite as mds
from mdsuite.transformations.map_molecules import MolecularMap


if __name__ == '__main__':

    filename = 'dump.lammpstrj'
    project = mds.Project(name='sdf_test')
    project.add_experiment(
        experiment='bmim_bf4',
        timestep=0.1,
        temperature=300.0,
        units='real',
        cluster_mode=False
    )

    # #project.experiments.bmim_bf4.perform_transformation("UnwrapCoordinates")
    # mapper = MolecularMap(
    #     project.experiments.bmim_bf4,
    #     molecules={
    #         'bmim-7': {'smiles': 'CCCCN1C=C[N+](+C1)C', 'cutoff': 1.9},
    #         'bf4-7': {'smiles': '[B-](F)(F)(F)F', 'cutoff': 2.4}
    #               }
    # )
    # mapper.run_transformation()
    project.run_computation.SpatialDistributionFunction(species=['bmim-7', 'bf4-7'],
                                                        r_min=1.0,
                                                        r_max=3.0,
                                                        number_of_configurations=5,
                                                        start=0,
                                                        stop=40)

    project.experiments.bmim_bf4.run_visualization()
