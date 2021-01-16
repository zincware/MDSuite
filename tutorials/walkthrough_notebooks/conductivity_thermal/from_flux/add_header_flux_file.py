# This script is far from ideal...but it is only to give the structure.

import re
import os
from shutil import copyfile
import time


def prepend(filename, data, bufsize=1<<15):
    # adapted from https://stackoverflow.com/questions/5287762/how-to-insert-a-new-line-before-the-first-line-in-a-file-using-python/5290636#5290636
    # backup the file
    backupname = filename + os.extsep+'bak'
    try: os.unlink(backupname) # remove previous backup if it exists
    except OSError: pass
    os.rename(filename, backupname)

    # open input/output files,  note: outputfile's permissions lost
    with open(backupname) as inputfile, open(filename, 'w') as outputfile:
        # prepend
        outputfile.write(data)
        # copy the rest (remove the first line)
        _ = inputfile.readline()
        buf = inputfile.read(bufsize)
        while buf:
            outputfile.write(buf)
            buf = inputfile.read(bufsize)

    # remove backup on success
    try: os.unlink(backupname)
    except OSError: pass

def find_match(lammpsfilename, match):
    # find match in the line given a list of words as hint
    with open(lammpsfilename,'r') as fp:
        line_with_match = []
        for line in fp:
            if all(x in line for x in match):
                return line

def get_property(line_match):
    # this takes the value of a dict (usually a string) and returns a value of the property.
    # it is built this way to handle the possible errors.
    try:
        property_extracted = float(re.findall('\d*\.?\d+', line_match)[0])
    except IndexError:
        property_extracted = line_match.split()[1]
    except TypeError:
        property_extracted = None
    return property_extracted


def read_data(lammpsfilename):

    # inputs to match (some variables and the line of headers)
    things_to_match = {'density':           ['variable','Densi', 'equal'],
                       'atom_side':         ['variable','atom_side', 'equal'],
                       'dt':                ['variable', 'dt', 'equal'],
                       'units':             ['units'],
                       'molarMass':         ['variable', 'molarMass', 'equal'],
                       'pair_style':        ['pair_style'],
                       'timesteps_conduct': ["variable", 'timesteps_conductivity', 'equal'],
                       'output_step':       ['variable','output_step', 'equal'],
                       'temperature':       ['variable', 'T', 'equal'],
                       'dt_production': ['variable', 'dt_production', 'equal'],
                       'volume': ['the', 'volume', 'of', 'the', 'box', 'is:']
                       }

    line_fix_print_headers = "$(c_flux_thermal[1]) $(c_flux_thermal[2]) $(c_flux_thermal[3])"

    # match the different properties required

    matches = dict.fromkeys(things_to_match.keys())
    properties_extracted = dict.fromkeys(things_to_match.keys())

    for property, indicators in things_to_match.items():
        matches[property] = find_match(lammpsfilename, indicators)
        properties_extracted[property] = get_property(matches[property])
        if properties_extracted[property] is None:
            print(f'"{property}" not found, will be printed as "nan"')
            properties_extracted[property] = -1

    properties_extracted['num_atoms'] = properties_extracted["atom_side"]**3


    # get the header line
    line_header = find_match(lammpsfilename, line_fix_print_headers)
    line_header = line_header.split(sep='"')[1].replace("$","").replace("(","").replace(")","")

    text_to_replace = (f'# Fix print output for fix print_data\n'
                       f'# {properties_extracted["pair_style"]} potential \n'
                       f'# {properties_extracted["num_atoms"]} atoms, dens. {properties_extracted["density"]}g/cm^3 \n'
                       f'# dt = {properties_extracted["dt"]}, {properties_extracted["timesteps_conduct"]} timesteps, output_step = every {properties_extracted["output_step"]} step \n'
                       f'# Temperature = {properties_extracted["temperature"]} K, V = {properties_extracted["volume"]} A3 \n'
                       f'# LAMMPS {properties_extracted["units"]} units \n'
                       f'{line_header} \n')
    return text_to_replace

def check_header_existance(filename):
    with open(filename,'r') as fp:
        next(fp) # skip the first line
        head = [next(fp) for x in range(2)]
    if all([line.startswith("# ") for line in head]):
        print('It seems the header already exists, I will not add it.')
        return True
    else:
        return False

def add_header(filename, lammpsfilename, keep_copy=True):

    # Check if header already exists
    header_exists = check_header_existance(filename=filename)

    if not header_exists:
        if keep_copy:
            copyfile(filename, filename+'.backup')
            print('back copy done')

        text_to_replace = read_data(lammpsfilename)
        prepend(filename, text_to_replace)
        print("Job's Done! header replaced")

if __name__=='__main__':
    add_header('../flux_1.lmp_flux', '../argon.lammps')