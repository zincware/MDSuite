import numpy as np
import matplotlib.pyplot as plt
import os
import MDAnalysis as MD

os.chdir('{0}K_Analysis'.format(1300))  # Change to correct directory
box_array = np.array([32.65, 32.65, 32.65])
species_list = ['Na', 'Cl']
positions_matrix = []
for species in species_list:
    positions_matrix.append(np.load('{0}_Positions.npy'.format(species)))

def Center_Box():
    """ Center atoms in box """

    for i in range(len(species_list)):
        for j in range(len(positions_matrix[0])):
            positions_matrix[i][j] -= (box_array/2)

def Unwrap():
    """ Unwrap the box """

    Unwrapped_matrix = []
    Center_Box()
    differences_matrix = []
    for i in range(len(species_list)):
        differences_matrix.append(np.diff(positions_matrix[i], axis=0))

    Unwrapped_matrix.append(positions_matrix[0][0][0])
    for i in range(len(positions_matrix[0][0])):
        displacement = []
        if abs(differences_matrix[0][0][i][0]) > box_array[0]-5:
            displacement.append(differences_matrix[0][0][i][0] - np.sign(differences_matrix[0][0][i][0])*(box_array[0]))
        else:
            displacement.append(differences_matrix[0][0][i][0])
        if abs(differences_matrix[0][0][i][1]) > box_array[1]-5:
            displacement.append(differences_matrix[0][0][i][1] - np.sign(differences_matrix[0][0][i][1])*(box_array[0]))
        else:
            displacement.append(differences_matrix[0][0][i][1])
        if abs(differences_matrix[0][0][i][2]) > box_array[2]-5:
            displacement.append(differences_matrix[0][0][i][2] - np.sign(differences_matrix[0][0][i][2])*(box_array[0]))
        else:
            displacement.append(differences_matrix[0][0][i][2])

        print(displacement)
        Unwrapped_matrix.append(np.array(Unwrapped_matrix[i]) + np.array(displacement))




Unwrap()

