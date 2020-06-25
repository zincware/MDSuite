import numpy as np
from itertools import islice
import os
import timeit
import dask.dataframe as da

#filename = "test_simulatin.xyz"

filename = "/beegfs/work/stovey/LAMMPSSims/NaCl/scaledSim/Thesis_Sims/1300K/NaCl_Velocities.xyz"
species_summary = {}
dimensions = 3

def Get_System_Properties():
    with open(filename) as f: 
        
        # Get the number of lines
        for number_of_lines, l in enumerate(f):
            pass

        f.seek(0) # Return to start of file
        
        number_of_atoms = int(list(islice(f, 4))[3]) # Calculate the number of atoms
        number_of_configurations = int((number_of_lines + 1)/(number_of_atoms + 9))
        
        # Get the number of species and their indices (positions in the data file)
        i = 0
        for line in list(islice(f, 5, 5 + number_of_atoms)):
            if line.split()[2] not in species_summary:
                species_summary[line.split()[2]] = []

            species_summary[line.split()[2]].append(i)
            i += 1

        #f.seek(0)
        #initial_position = 9
        #final_position = number_of_lines + 1 + 9


        #data = list(islice(f, initial_position, final_position, number_of_atoms + 9))


        #test = map(lambda str: str.split(), data)
        #print(np.array(list(test))[:, 3])
        #data = np.array([data[i].split() for i in range(len(data))])
        #print(data[:, 3])
        #for i in range(len(data)):
        #    data[i] = data[i].split()
        #print(np.shape(data))
        #new_data = []
        #for i in range(len(data)):
            #new_data.append(data[i].split())

        f.seek(0)
        for i in range(len(list(species_summary))):
            prop_x = []
            print("i: {0}".format(i))
            for index in species_summary[list(species_summary)[i]]:
                print("index: {0}".format(index))
                initial_position = 9 + index
                final_position = number_of_lines + 1 + 9 + index
                data = list(islice(f, initial_position, final_position, number_of_atoms + 9))
                #data = f[initial_position, final_position, number_of_atoms + 9]
                data = map(lambda str: str.split(), data)
                prop_x.append(np.array(list(data))[:, 3])
                f.seek(0)



        
        # Generate the property matrices and save
        #for i in range(len(list(species_summary))):
            #for index in species_summary[list(species_summary)[i]]:
                #pass


    #print(species_summary)
    #print(number_of_atoms)
    #print(number_of_configurations)
    #print(final_position)

execution_time = timeit.timeit('Get_System_Properties()', 'from __main__ import Get_System_Properties', number=1)
print(execution_time)


#np.array(data_array[index::1 * (self.number_of_atoms + 2)])[:,1 + i*self.dimensions].astype(float)[:, None]
