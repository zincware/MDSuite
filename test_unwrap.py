import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as MD
from MDAnalysis import transformations

u = MD.Universe("/beegfs/work/stovey/LAMMPSSims/NaCl/scaledSim/Thesis_Sims/1300K/rerun/NaCl_Velocities.lammpsdump",
                atom_style="id type element x y z")

ag=u.atoms


