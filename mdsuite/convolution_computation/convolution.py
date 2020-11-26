from tqdm import tqdm
import numpy as np
from scipy import signal

def convolution(loop_range, flux, data_range, time):
    sigma = np.empty((loop_range,))
    # main loop for computation
    for i in tqdm(range(loop_range)):
        jacf = (signal.correlate(flux[:, 0][i:i + data_range],
                                  flux[:, 0][i:i + data_range],
                                  mode='full', method='fft') +
                 signal.correlate(flux[:, 1][i:i + data_range],
                                  flux[:, 1][i:i + data_range],
                                  mode='full', method='fft') +
                 signal.correlate(flux[:, 2][i:i + data_range],
                                  flux[:, 2][i:i + data_range],
                                  mode='full', method='fft'))

        # Cut off the second half of the acf
        jacf = jacf[int((len(jacf) / 2)):]
        # if self.plot:
        #     averaged_jacf += jacf

        integral = np.trapz(jacf, x=time)
        sigma[i] = integral

    return sigma