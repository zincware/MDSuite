import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
try:
    import thermocepstrum as tc
except ImportError:
    from sys import path
    path.append('..')
    import thermocepstrum as tc

c = plt.rcParams['axes.prop_cycle'].by_key()['color']

jfile = tc.i_o.TableFile('../flux_1_cepstral.lmp_flux', group_vectors=True)

jfile.read_datalines(start_step=0, NSTEPS=0, select_ckeys=['flux', 'temp'])

DT_FS = 2                # time step [fs]
TEMPERATURE = np.mean(jfile.data['temp'])   # temperature [K]
VOLUME = 66241.9672397342    # volume [A^3]

j = tc.HeatCurrent(jfile.data['flux'], units='real', DT_FS=DT_FS, TEMPERATURE=TEMPERATURE, VOLUME=VOLUME)

# Periodogram with given filtering window width
ax = j.plot_periodogram(PSD_FILTER_W=0.005, kappa_units=True)
print(j.Nyquist_f_THz)
# plt.xlim([0, 1])
#ax[0].set_ylim([0, 150]);
# ax[1].set_ylim([10, 25]);

FSTAR_THZ = 10
jf, ax = j.resample(fstar_THz=FSTAR_THZ, plot=True, freq_units='thz')
# jf, ax = j.resample(fstar_THz=FSTAR_THZ, plot=True, freq_units='thz')   # for thermocepstrum-develop
# plt.xlim([0, 1])
# ax[1].set_ylim([12,18]);

jf.cepstral_analysis() #Kmin_corrfactor=2.)

# Cepstral Coefficients
# print('c_k = ', jf.dct.logpsdK)
ax = jf.plot_ck()
ax.set_xlim([0, jf.dct.aic_Kmin * 5])
ax.set_ylim([-0.5, 0.5])
ax.grid();

# AIC function
f = plt.figure()
plt.plot(jf.dct.aic, '.-', c=c[0])
plt.xlim([0, jf.dct.aic_Kmin * 4])
plt.ylim([jf.dct.aic_min * 0.99, jf.dct.aic_min * 1.1]);
plt.axvline(jf.dct.aic_Kmin, ls='--')

print('K of AIC_min = {:d}'.format(jf.dct.aic_Kmin))
print('AIC_min = {:f}'.format(jf.dct.aic_min))

# L_0 as a function of cutoff K
ax = jf.plot_L0_Pstar()
ax.set_xlim([0, jf.dct.aic_Kmin * 8])
# ax.set_ylim([-2, 0]);
ax.grid()

print('K of AIC_min = {:d}'.format(jf.dct.aic_Kmin))
print('AIC_min = {:f}'.format(jf.dct.aic_min))

# kappa as a function of cutoff K
ax = jf.plot_kappa_Pstar()
ax.set_xlim([0, jf.dct.aic_Kmin * 6])
ax.set_ylim([0, 400.]);

print('K of AIC_min = {:d}'.format(jf.dct.aic_Kmin))
print('AIC_min = {:f}'.format(jf.dct.aic_min))

print(jf.cepstral_log)

# filtered log-PSD
ax = j.plot_periodogram(0.0001, kappa_units=True)
ax = jf.plot_periodogram(0.0001, axes=ax, kappa_units=True)
ax = jf.plot_cepstral_spectrum(axes=ax, kappa_units=True)
ax[0].axvline(x = jf.Nyquist_f_THz, ls='--', c='r')
ax[1].axvline(x = jf.Nyquist_f_THz, ls='--', c='r')
# plt.xlim([0., 0.12])
# ax[1].set_ylim([10,25])
ax[0].legend(['original', 'resampled', 'cepstrum-filtered'])
ax[1].legend(['original', 'resampled', 'cepstrum-filtered']);


plt.show()
