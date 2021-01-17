import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = np.load('Na_einstein_diffusion_coefficients.npy', allow_pickle=True)
log_x = np.log10(data[0, 1:])
log_y = np.log10(data[1, 1:])


def func(x, a):
    return x + a


popt, pcov = curve_fit(func, log_x, log_y)

print(10**popt[0])


plt.plot(log_x, log_y, label='data')
plt.plot(log_x, func(log_x, *popt), label='fit')
plt.legend()
plt.show()
