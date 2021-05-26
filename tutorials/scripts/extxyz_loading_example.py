"""
Read an xyz file
"""
import mdsuite as mds  # Import the mdsuite python package

if __name__ == "__main__":

    Argon = mds.Experiment('Argon', storage_path='./', time_step=1.0, temperature=85, units='metal')

    Argon.add_data(trajectory_file='/beegfs/work/fzills/work/FNET-Models/Ar85K/Ar85K_q_SOAP_r6l5n5_16x16_efe_b25_1e'
                                   '-04_1000confs.extxyz',
                   file_format='extxyz')
    Argon.sample_rate = 10
    Argon.run_computation.RadialDistributionFunction(plot=True, number_of_configurations=500, start=1, stop=700)
