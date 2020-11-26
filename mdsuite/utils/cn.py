import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
from meta_functions import golden_section_search
from scipy.signal import find_peaks
from scipy.signal import savgol_filter



filepath = "/data/stovey/New/NaCl_1074_15K_GK/data/(0, 1)_radial_distribution_function.npy"

radii, rdf = np.load(filepath, allow_pickle=True)


test = savgol_filter(rdf, 17 , 2)

plt.plot(radii, test)
   
plt.legend()
plt.show()

minimum = golden_section_search([radii, rdf], 6, 3)

peaks, _ = find_peaks(test, height=1.1)

plt.plot([minimum[0], minimum[1]],[rdf[np.where(radii == minimum[0])], rdf[np.where(radii == minimum[1])]], 'o')
plt.plot(radii[peaks], rdf[peaks], '+')
plt.plot(radii, rdf)
plt.show()


