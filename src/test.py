import re

data_array = []
with open("/tikhome/stovey/work/PES/Molten_Salts/Chlorides/NaCl/MachineLearnedPotentials/GAP/train.xyz") as f:
    for line in f:
        data_array.append(line.split())



Get_EXTXYZ_Properties(data_array)