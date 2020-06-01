import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def Build_Data(filename):

    data_array = []
    with open(filename) as f:
        for line in f:
            data_array.append(line.split())

    number_of_atoms = 1000
    Positions = [[], []]
    for j in range(2):
        for i in range(2 + j*500, 502 + j*500):
            Positions[j].append([np.array(data_array[i::(number_of_atoms + 2)])[:, 1].astype(float),
                                        np.array(data_array[i::(number_of_atoms + 2)])[:, 2].astype(float),
                                        np.array(data_array[i::(number_of_atoms + 2)])[:, 3].astype(float)])
    np.save('1400_Positions.npy', Positions)
    return Positions

def Load_Data(saved_file):
    Positions = np.load(saved_file, allow_pickle=True)
    Positions = np.array(Positions) - 16.415
    return Positions

def Calculate_Dipoles(Positions):

def main():

    #Positions = Build_Data('/beegfs/work/stovey/LAMMPSSims/NaCl/scaledSim/Thesis_Sims/1350K/Unwrapped.xyz')
    Positions = Load_Data('1400_Positions.npy')
    Dipole_Moments, subtractor = Calculate_Dipoles(Positions)
    msd, time = Calculate_MSD(Dipole_Moments, subtractor)
    Get_Conductivity(msd, time)


if __name__ == "__main__":
    main()
