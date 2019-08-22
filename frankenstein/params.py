import numpy as np
import matplotlib.pyplot as plt
import os
pwd = os.getcwd()
from scipy.interpolate import interp1d

from constants import rad_to_arcsec, deg_to_rad, sterad_to_arsec, c

disc = 'as209' # disc name (sets save directory for fit results)
#disc = 'elias24'
#disc = 'gwlup'
#disc = 'hd142666'
#disc = 'hd143006'
#disc = 'hd163296'
#disc = 'rulup'
#disc = 'sr4'
#disc = 'sz114'
#disc = 'sz129'

#disc = 'bp_tau'
#disc = 'do_tau' # TODO: keep?
#disc = 'dr_tau'

#disc = 'dsharp_5e10_C437_40minuteintegration_rev3'

#disc = 'synth_richard_profile_original'
#disc = 'synth_richard_profile_for_clean_comparison13'

#disc = 'synth_gauss_rev3'
#disc = 'asym_gauss_ring_rev3'
#disc = 'double_gauss_rev3'

#disc = 'gauss_ring_pt025_rev3'
#disc = 'gauss_ring_pt0125_briggs_negative2_rev3'

savedir = pwd + '/results/dsharp/%s/' % disc # full path to save directory
#savedir = pwd + '/results/taurus/%s/' % disc # full path to save directory # TODO: set savedir to input uvtable filename
#savedir = pwd + '/results/synthetic/%s/' % disc # full path to save directory
#savedir = pwd + '/results/real_obs_misc/%s/' % disc # full path to save directory

#uvtable_filename = ['as209_sc_cont_nofl_chnavg.txt'] # input uv table(s). if >1 provided, they will be concatenated # davide's data
#uvtable_filename = ['dsharp_5e10_C437_rev3.alma.cycle6.7.noisy.ms_uvtable.txt']

uvtable_filename = ['AS209_continuum_nopnt_nofl_HwuUXC4.npz'] # dsharp: as209
#uvtable_filename = ['Elias24_continuum_nopnt_nofl_FbwfzSt.npz']
#uvtable_filename = [''] # gwlup
#uvtable_filename = ['HD142666_continuum_nopnt_nofl_SAbhJvF.npz']
#uvtable_filename = ['HD143006_continuum_nopnt_nofl_DqKIJWO.npz']
#uvtable_filename = ['HD163296_continuum_nopnt_nofl_GlOXRTg.npz']
#uvtable_filename = ['RULup_continuum_nopnt_nofl_NTlBhwg.npz'] # dsharp: rulup
#uvtable_filename = ['SR4_continuum_nopnt_nofl_kRhVD8W.npz']
#uvtable_filename = ['Sz114_continuum_nopnt_nofl_XU3E1OT.npz']
#uvtable_filename = ['Sz129_continuum_nopnt_nofl_NVYRQav.npz']

#uvtable_filename = ['BP_Tau_selfcal_cont_bin60s_1chan.nopnt_HHIVVDe.npz'] # taurus: bp tau
#uvtable_filename = ['DO_Tau_selfcal_cont_bin60s_1chan_nopnt_nofl_QvuwlR4.npz']
#uvtable_filename = ['DR_Tau_selfcal_cont_bin60s_1chan_nopnt_nofl_Qz0ivzk.npz'] # taurus: dr tau

#wle = 0.00130344546957 # observing wavelength. unit: [m]
wle = 1. # dsharp: as209 (also all DSHARP and Taurus discs that MT reduced; he's already normalized u & v in [m] by each indiv. point's (c / freq))
freq = 239e9 # dsharp: many discs e.g. ru lup
#freq = 231.9e9 # dsharp: hd142666, elias24

dist = 121. # dsharp: as209
#dist = 136. # dsharp: elias24
#dist = 155. # dsharp: gwlup
#dist = 148. # dsharp: hd142666
#dist = 165. # dsharp: hd143006
#dist = 101. # dsharp: hd163296
#dist = 159. # dsharp: rulup
#dist = 134. # dsharp: sr4
#dist = 162. # dsharp: sz114
#dist = 161. # dsharp: sz129

use_freqs = False # True to use freqs in the UVTable rather than a single wavelength (wle) to normalize u and v distances
input_kl = False # True if the uvtable's u and v distances (normalized) are in k\lambda

Rmax = 1.7 # outer disc edge for real space fit. unit: [arcsec] # dsharp: as209
#Rmax = 1.5 # TODO: is this really setting Rmax for the fit?
#Rmax = 1.2
#Rmax = .7
#Rmax = .6 # dsharp; RU Lup
#Rmax = .5
#Rmax = 1. # dsharp: sz129

known_weights = True # False if the weights on the visibilites aren't known (in which case we coarsely estimate them)
wcorr = 1. # TODO: move this from param file to internal for "if known_weights = True"

known_geometry = True # True if inc, PA, dRA, dDec are provided; set False to fit for these
#inc, pa, dra, ddec = 0., 0., 0., 0. # disc inclination, position angle, \Delta right ascension, \Delta declination. unit: [deg, deg, arcsec, arcsec]
# TODO: if known_geometry = False, don't pass these in

#inc, pa, dra, ddec = 0.61286176 * 180 / np.pi, 1.50132798 * 180 / np.pi, 0.0009164178622590247 / rad_to_arcsec, -0.00024897169943341544 / rad_to_arcsec
#inc, pa, dra, ddec = 34.97, 85.76, -1.9 / 1e3 / rad_to_arcsec, 2.5 / 1e3 / rad_to_arcsec

#inc, pa, dra, ddec = 34.883, 85.674, -1.699 / 1e3, 3.102 / 1e3 # dsharp: as209
inc, pa, dra, ddec = 34.97, 85.76, -1.9 / 1e3, 2.5 / 1e3 # dsharp: as209, dsharp paper values
#inc, pa, dra, ddec = 33.932966683923155, 86.46996053622686, 0.0009163953052281576, -0.0002489963254808658 # dsharp: as209, gauss fit values
#inc, pa, dra, ddec = 17.52971266474803, -60.413303134503174, 0.014215884880306657, -0.08530798146760972 # dsharp: rulup, gauss fit values
#inc, pa, dra, ddec = 17.529764134641674, 119.5867782742138, 0.014215870802985668, -0.08530800176990953 # dsharp: rulup, gauss fit values if initial guess on PA = 90 deg

#inc, pa, dra, ddec = 38.2, 151.1, -.05, .09 # taurus: bp tau
#inc, pa, dra, ddec = 27.6, 170., 0.11, 0.19 # taurus: do tau. doesn't seem to be right
#inc, pa, dra, ddec = 5.4, 3.4, .09, -.04 # taurus: dr tau

cutuv = False # True to cut the data at some maximum uv-baseline
maxuv = 4.5e6 # baseline at which to cut. unit: [\lambda] # TODO: add option to cut at minuv (and in deproject_funcs)
set_im_to_0 = False # True to set Im(V) data to 0

add_noise = False
noise_ratio = 5.

#nbins = np.int(Rmax * 198 / 1.7) # number of bins in real space fit (equivalently the number of Bessel functions in Fourier domain fit)
nbins=128
alpha = 1.05 #1.1 # alpha parameter for smoothing GP power spectrum # TODO: 1.1 seems to perform better
# unless the low SNR data at high baseline is high V
smooth_strength = 1e-4 #1e-4 # strength of smoothing
#nbins_list = [128]
#alphas = [1.1, 1.5]
#smooth_strengths = [1e-4, 1e-1]#, 1e-4, 1e-2, 1]
enforce_positivity = False # True to evaluate the 'mean' fit function as the best fit subject to the constraint that the reconstructed function is positive semi-definite

find_eff_res = True # whether to estimate the data's effective resolution by refitting while neglecting a successively larger portion of the longest baselines

save_fit_values = True # True to save fit values to .txt file
plot_results = False # True to generate and save plots for fit
ms_figs = False
ms_f1 = False
ms_f2 = False
ms_f3 = False
ms_f4 = False
ms_f5 = True
cmaps = [plt.cm.cool, plt.cm.autumn_r, plt.cm.winter_r, plt.cm.copper_r] # color map for power spectrum evolution plot
uvbin_size = 1e3 # bin size for uvplot. unit: [\lambda]
uvbin_size2 = 50e3
snr_plot_binwidth = 1e3 # bin size for plot of data SNR. unit: [\lambda]
snr_plot_binwidth2 = 50e3

image_extracted_profile = False # to provide a real space profile extracted from the disc image for comparison to fit
known_profile = True # True to provide an input real space profile (for plotting and computing fit residuals)
#model = [] # function form of provided input real space profile
def model(r):
    #model = InterpolatedDSHARPProfile('/Users/morgan/gdrive_jmjenn/recipes/Frankenstein/uvtables/as209_dsharp_profile.txt',frequency=239e9, norm=1)
    model = InterpolatedDSHARPProfile('/Users/morgan/gdrive_jmjenn/recipes/Frankenstein/uvtables/%s_dsharp_profile.txt'%disc, frequency=freq, norm=1, dist_correct=121./dist)
    return model(r)

class InterpolatedDSHARPProfile(object):
    def __init__(self, data_file, frequency, norm=1, dist_correct=1.0):
        data = np.genfromtxt(data_file, unpack=True)

        r, T = data[1], data[4]

        r *= dist_correct

        wave = c / frequency

        I = 2 * 1.3806503e-23 * T  / wave**2
        I *= 1e26 / norm

        self._Inu = interp1d(r, I, fill_value = (I[0], 0), bounds_error=False)

    def __call__(self, r):
        return self._Inu(r * rad_to_arcsec)
#'''
if savedir == pwd + '/results/synthetic/%s/' % disc and disc == 'as209':
    inc, pa, dra, ddec, wle, wcorr = [0.61286176 * 180 / np.pi, 1.50132798 * 180 / np.pi, 1.339188612881619e-06,
                                          1.65779981e-07 * 180 / np.pi, 0.00130347953005, 1.]
    #Rmax = 1.6
    #Rmax = 1.35

    #uvtable_filename = ['as209_sc_cont_nofl_chnavg.txt']
#'''
if disc == 'synth_dsharp' or disc == 'dsharp_rev3' or disc == 'dsharp_5e10_rev3' or disc == 'dsharp_C437_rev3' \
        or disc == 'dsharp_5e10_C437_rev3' or disc == 'dsharp_5e10_C437_40minuteintegration_rev3':
    inc, pa, dra, ddec, wle = [0., 0., 0., 0., 0.00130344546957]
    #Rmax = 1.5
    uvtable_filename = ['synth_dsharp_profile_inc0_B6_C436.alma.cycle6.6.noisy.ms_uvtable.txt']

    if disc == 'dsharp_rev3':
        #Rmax = 1.35
        uvtable_filename = ['dsharp_rev3.alma.cycle6.6.noisy.ms_uvtable.txt']

    if disc == 'dsharp_5e10_rev3':
        #Rmax = 1.35
        uvtable_filename = ['dsharp_5e10_rev3.alma.cycle6.6.noisy.ms_uvtable.txt']

    if disc == 'dsharp_C437_rev3':
        #Rmax = 1.35
        uvtable_filename = ['dsharp_C437_rev3.alma.cycle6.7.noisy.ms_uvtable.txt']

    if disc == 'dsharp_5e10_C437_rev3':
        #Rmax = 1.35
        uvtable_filename = ['dsharp_5e10_C437_rev3.alma.cycle6.7.noisy.ms_uvtable.txt']

    if disc == 'dsharp_5e10_C437_40minuteintegration_rev3':
        #Rmax = 1.5
        uvtable_filename = ['dsharp_5e10_C437_40minuteintegration_rev3.alma.cycle6.7.noisy.ms_uvtable.txt']

    def model(r):
        normalization = 5e10
        if disc == 'dsharp_rev3' or disc == 'dsharp_C437_rev3': normalization = 1e10
        from scipy.interpolate import interp1d
        rs, bs = np.genfromtxt('/Users/morgan/gdrive_jmjenn/recipes/Frankenstein/uvtables/dsharp_as209.csv', delimiter=',').T
        rs /= 121.
        bs = bs / max(bs) * normalization
        interp_f = interp1d(rs / rad_to_arcsec, bs, kind='linear', bounds_error=False, fill_value=(bs[0], bs[-1]))
        return interp_f(r)

if disc == 'synth_richard_profile_original' or disc == 'synth_richard_profile_original_5e10':
    inc, pa, dra, ddec, wle = [0., 0., 0., 0., 0.00130344546957]
    Rmax = 3.
    uvtable_filename = ['synth_richard_profile_original.alma.cycle6.6.noisy.ms_uvtable.txt']
    if disc == 'synth_richard_profile_original_5e10':
        uvtable_filename = ['synth_richard_profile_original_5e10.alma.cycle6.6.noisy.ms_uvtable.txt']

    def model(r):
        x = r * rad_to_arcsec
        env = np.exp(- 0.5*x**2)
        gap1 = 0.7 * ((x > 0.1) & (x < 0.2))
        gap2 = 0.2  * ((x > 0.7) & (x < 1.0 ))
        gap3 = 0.15 * ((x > 0.9) & (x < 1.0 ))
        ring1 = 0.5 * ((x > 1.0) & (x < 1.2))
        if disc == 'synth_richard_profile_original_5e10': return 5e10 * env * (1 - gap1 - gap2 - gap3 + ring1)
        return .5e10 * env * (1 - gap1 - gap2 - gap3 + ring1)

        return interp_f(r)

if disc == 'synth_richard_profile_pt1_smallest':
    inc, pa, dra, ddec, wle = [0., 0., 0., 0., 0.00130344546957]
    Rmax = 3.
    uvtable_filename = ['synth_richard_profile_inc0_B6_C436.alma.cycle6.6.noisy.ms_uvtable.txt']

    def model(r):
        x = r * rad_to_arcsec
        env = np.exp(- 0.5*x*x)
        gap1 = 0.7  * ((x > 0.1) & (x < 0.2 ))
        gap2 = 0.2  * ((x > 0.7) & (x < 1.0 ))
        gap3 = 0.15 * ((x > 0.9) & (x < 1.0 ))
        ring1 = 0.5 * ((x > 1.0) & (x < 1.2))
        return 5e10 * env * (1 - gap1 - gap2 - gap3 + ring1)

if disc == 'synth_richard_profile_for_clean_comparison13' or disc == 'synth_richard_profile_for_clean_comparison13_uniform_weighting':
    inc, pa, dra, ddec, wle = [0., 0., 0., 0., 0.00130344546957]
    Rmax = 1.75
    uvtable_filename = ['synth_richard_profile_for_clean_comparison13.alma.cycle6.6.noisy.ms_uvtable.txt']
    if disc == 'synth_richard_profile_for_clean_comparison13_uniform_weighting':
        uvtable_filename = ['synth_richard_profile_for_clean_comparison13_uniform_weighting.alma.cycle6.6.noisy.ms_uvtable.txt']

    def model(r):
        x = r * rad_to_arcsec
        gap1 = 0.7 * ((x > 0.1) & (x < 0.2))
        env = np.exp(- 0.5 * x ** 6)
        gap2 = 0.2 * ((x > 0.4) & (x < .7))
        gap3 = 0.15 * ((x > 0.6) & (x < .7))
        ring1 = 0.5 * ((x > .9) & (x < 1.))
        gap4 = .99999 * ((x > 1.2))
        return .5e10 * env * (1 - gap1 - gap2 - gap3 - gap4 + ring1)


    def richard_model(r):
        x = r / 1.
        # env = np.exp(- 0.5*x**2)
        gap1 = 0.7 * ((x > 0.1) & (x < 0.2))
        # gap2 = 0.2  * ((x > 0.7) & (x < 1.0 ))
        # gap3 = 0.15 * ((x > 0.9) & (x < 1.0 ))
        # ring1 = 0.5 * ((x > 1.0) & (x < 1.2))
        # return env * (1 - gap1 - gap2 - gap3 + ring1)

        env = np.exp(- 0.5 * x ** 6)
        gap2 = 0.2 * ((x > 0.4) & (x < .7))
        gap3 = 0.15 * ((x > 0.6) & (x < .7))
        ring1 = 0.5 * ((x > .9) & (x < 1.))
        return 5e10 * env * (1 - gap1 - gap2 - gap3 + ring1)

if disc == 'double_gauss_rev3':
    inc, pa, dra, ddec, wle = [0., 0., 0., 0., 0.00130344546957]
    Rmax = 1.2
    uvtable_filename = ['double_gauss_rev3.alma.cycle6.6.noisy.ms_uvtable.txt']

    def model(r):
        normalization = 1e10

        def gauss(x, a, b, c, d):  # a = peak mag., b = x-val at peak, c ~ bell width, d = y-offset
            return a * np.exp(-(x - b) ** 2 / (2. * c ** 2)) + d

        def sigmoid(x, a, c1, c2, d, middle):
            half1 = gauss(x, a, middle, c1, d) * ((x <= middle))
            half2 = gauss(x, a, middle, c2, d) * ((x > middle))
            return half1 + half2
            # return a * np.exp(-(x - middle)**2 / (2. * c1**2)) + d

        return gauss(r, normalization, 0 / rad_to_arcsec, .2 / rad_to_arcsec, 0) + \
               sigmoid(r, normalization, .1 / rad_to_arcsec, .2 / rad_to_arcsec, 0., .6 / rad_to_arcsec)

if disc == 'synth_gauss' or disc == 'synth_gauss_reverse' or disc == 'synth_gauss_rev3':
    inc, pa, dra, ddec, wle = [0., 0., 0., 0., 0.00130344546957]
    #Rmax = .7
    Rmax = 1.2
    uvtable_filename = ['synth_gauss_inc0_B6_C436.alma.cycle6.6.noisy.ms_uvtable.txt']
    if disc == 'synth_gauss_rev3':
        uvtable_filename = ['synth_gauss_rev3.alma.cycle6.6.noisy.ms_uvtable.txt']

    def gauss(x, a, b, c, d):  # a = peak mag., b = x-val at peak, c ~ bell width, d = y-offset
        return a * np.exp(-(x - b) ** 2 / (2. * c ** 2)) + d

    def model(r):
        #x = r * rad_to_arcsec
        normalization = 5e10
        if disc == 'synth_gauss_rev3':
            normalization = 1e10
        #R = np.linspace(Rin, Rin + dR * nR, nR, endpoint=False)
        return gauss(r, normalization, 0 / rad_to_arcsec, .2 / rad_to_arcsec, 0)

if disc == 'asym_gauss_ring_test' or disc == 'asym_gauss_ring_rev3':
    inc, pa, dra, ddec, wle = [0., 0., 0., 0., 0.00130344546957]
    Rmax = 1.2
    #uvtable_filename = ['asym_gauss_ring11.alma.cycle6.6.noisy.ms_uvtable.txt']
    #uvtable_filename = ['asym_gauss_ring2.alma.cycle6.6.noisy.ms_uvtable.txt']
    #uvtable_filename = ['asym_gauss_ring3.alma.cycle6.6.noisy.ms_uvtable.txt']
    #uvtable_filename = ['asym_gauss_ring4.alma.cycle6.6.noisy.ms_uvtable.txt']
    #uvtable_filename = ['asym_gauss_ring5.alma.cycle6.6.noisy.ms_uvtable.txt']
    #uvtable_filename = ['asym_gauss_ring7.alma.cycle6.6.noisy.ms_uvtable.txt']
    #uvtable_filename = ['asym_gauss_ring8.alma.cycle6.6.noisy.ms_uvtable.txt']
    #uvtable_filename = ['asym_gauss_ring9.alma.cycle6.6.noisy.ms_uvtable.txt']
    #uvtable_filename = ['asym_gauss_ring10.alma.cycle6.6.noisy.ms_uvtable.txt']
    uvtable_filename = ['asym_gauss_ring_rev3.alma.cycle6.6.noisy.ms_uvtable.txt']

    def gauss(x, a, b, c, d):  # a = peak mag., b = x-val at peak, c ~ bell width, d = y-offset
        return a * np.exp(-(x - b) ** 2 / (2. * c ** 2)) + d

    def sigmoid(x, a, c1, c2, d, middle):
        half1 = gauss(x, a, middle, c1, d) * ((x <= middle))
        half2 = gauss(x, a, middle, c2, d) * ((x > middle))
        return half1 + half2

    def model(r):
        normalization = 5e10
        if disc == 'asym_gauss_ring_rev3':
            normalization = 1e10
            return sigmoid(r, normalization, .1 / rad_to_arcsec, .2 / rad_to_arcsec, 0., .6 / rad_to_arcsec)
        return sigmoid(r, normalization, .2 / rad_to_arcsec, .1 / rad_to_arcsec, 0., .7 / rad_to_arcsec)
        #return gauss(r, normalization, .5 / rad_to_arcsec, .1 / rad_to_arcsec, 0.)

if disc == 'synth_gauss_ring' or disc == 'synth_gauss_ring_moreruns' or disc == 'synth_gauss_ring_fwhm_pt1_test' \
        or disc == 'synth_gauss_ring_fwhm_pt05' or disc == 'synth_gauss_ring_fwhm_pt025' or disc == 'synth_gauss_ring_fwhm_pt025_small_rout' \
        or disc == 'synth_gauss_ring_fwhm_pt0125' or disc == 'synth_gauss_ring_test' or disc == 'gauss_ring_pt0125_rev3' or disc == 'gauss_ring_pt025_rev3' \
        or disc == 'gauss_ring_pt0125_briggs_negative2_rev3':
    inc, pa, dra, ddec, wle = [0., 0., 0., 0., 0.00130344546957]

    Rmax = 2.
    if disc == 'synth_gauss_ring_fwhm_pt1_test': Rmax = 1.
    if disc == 'synth_gauss_ring_fwhm_pt05': Rmax = 1.
    if disc == 'synth_gauss_ring_fwhm_pt025': Rmax = 1.
    if disc == 'synth_gauss_ring_fwhm_pt025_small_rout': Rmax = .7
    if disc == 'synth_gauss_ring_fwhm_pt0125': Rmax = 1.
    if disc == 'synth_gauss_ring_test': Rmax = 3.

    uvtable_filename = ['synth_gauss_ring_inc0_B6_C436.alma.cycle6.6.noisy.ms_uvtable.txt']
    if disc == 'synth_gauss_ring_fwhm_pt1_test': uvtable_filename = ['synth_gauss_ring_fwhm_pt1.alma.cycle6.6.noisy.ms_uvtable.txt']
    if disc == 'synth_gauss_ring_fwhm_pt05': uvtable_filename = ['synth_gauss_ring_fwhm_pt05.alma.cycle6.6.noisy.ms_uvtable.txt']
    if disc == 'synth_gauss_ring_fwhm_pt025' or disc == 'synth_gauss_ring_fwhm_pt025_small_rout': uvtable_filename = ['synth_gauss_ring_fwhm_pt025.alma.cycle6.6.noisy.ms_uvtable.txt']
    if disc == 'synth_gauss_ring_fwhm_pt0125': uvtable_filename = ['synth_gauss_ring_fwhm_pt0125.alma.cycle6.6.noisy.ms_uvtable.txt']
    if disc == 'synth_gauss_ring_test': uvtable_filename == ['synth_gauss_ring_inc0_B6_C436_test.alma.cycle6.6.noisy.ms_uvtable.txt']

    def gauss(x, a, b, c, d):  # a = peak mag., b = x-val at peak, c ~ bell width, d = y-offset
        return a * np.exp(-(x - b) ** 2 / (2. * c ** 2)) + d

    def model(r):
        #x = r * rad_to_arcsec
        normalization = 5e10
        #R = np.linspace(Rin, Rin + dR * nR, nR, endpoint=False)
        return gauss(r, normalization, 1 / rad_to_arcsec, .2 / rad_to_arcsec, 0)

    if disc == 'synth_gauss_ring_fwhm_pt1_test':
        def model(r):
            # x = r * rad_to_arcsec
            normalization = 5e10
            # R = np.linspace(Rin, Rin + dR * nR, nR, endpoint=False)
            return gauss(r, normalization, .5 / rad_to_arcsec, .1 / rad_to_arcsec, 0)

    if disc == 'synth_gauss_ring_fwhm_pt05':
        def model(r):
            normalization = 5e10
            return gauss(r, normalization, .5 / rad_to_arcsec, .05 / rad_to_arcsec, 0)

    if disc == 'synth_gauss_ring_fwhm_pt025' or disc == 'synth_gauss_ring_fwhm_pt025_small_rout':
        def model(r):
            normalization = 5e10
            return gauss(r, normalization, .5 / rad_to_arcsec, .025 / rad_to_arcsec, 0)

    if disc == 'synth_gauss_ring_fwhm_pt0125':
        def model(r):
            normalization = 5e10
            return gauss(r, normalization, .5 / rad_to_arcsec, .0125 / rad_to_arcsec, 0)

    if disc == 'synth_gauss_ring_test':
        def model(r):
            normalization = 5e10
            return gauss(r, normalization, 1. / rad_to_arcsec, .2 / rad_to_arcsec, 0)

    if disc == 'gauss_ring_pt0125_rev3':
        uvtable_filename = ['gauss_ring_pt0125_rev3.alma.cycle6.6.noisy.ms_uvtable.txt']
        Rmax = 1.
        def model(r):
            normalization = 1e10
            return gauss(r, normalization, .5 / rad_to_arcsec, .0125 / rad_to_arcsec, 0)

    if disc == 'gauss_ring_pt0125_briggs_negative2_rev3':
        uvtable_filename = ['gauss_ring_pt0125_briggs_negative2_rev3.alma.cycle6.6.noisy.ms_uvtable.txt']
        Rmax = 1.

        def model(r):
            normalization = 1e10
            return gauss(r, normalization, .5 / rad_to_arcsec, .0125 / rad_to_arcsec, 0)

    if disc == 'gauss_ring_pt025_rev3':
        uvtable_filename = ['gauss_ring_pt025_rev3.alma.cycle6.6.noisy.ms_uvtable.txt']
        Rmax = 1.
        def model(r):
            normalization = 1e10
            return gauss(r, normalization, .5 / rad_to_arcsec, .025 / rad_to_arcsec, 0)

"""
def model(r):
    from scipy.interpolate import interp1d
    '''
    rs, bs = np.genfromtxt('/Users/morgan/gdrive_jmjenn/recipes/Frankenstein/uvtables/dsharp_as209.csv', delimiter=',').T
    rs /= 121.
    #normalization = 2.8e10
    bs = bs / max(bs) #* normalization
    interp_f = interp1d(rs / rad_to_arcsec, bs, kind='linear', bounds_error=False, fill_value=(bs[0], bs[-1]))
    return interp_f(r)
    ''' # TODO: fix transformation for bs
    _, rs, bs, bs_err, _, _, _, _ = np.genfromtxt(pwd + '/uvtables/' + 'as209_dsharp_fit.txt', skip_header=1).T
    rs /= rad_to_arcsec
    bs = bs * (np.pi * 38 * 36 / (4 * np.log(2))) * sterad_to_arsec
    bs_err = bs_err * (np.pi * 38 * 36 / (4 * np.log(2))) * sterad_to_arsec
    interp_f = interp1d(rs, bs, kind='linear', bounds_error=False, fill_value=(bs[0], bs[-1]))
    return interp_f(r)
    #'''
"""

#disc = 'mock1_test'
#disc = 'mock1_uvtable_mod_fittedi30' # checked, same as mock1
#disc = 'mock1_highi'
#disc = 'mock2'
#disc = 'as209'
#disc = 'as209_test'
#disc = 'as209_test2'
#disc = 'as209_test_set_im_to_0' # set imaginary components of data to 0 to see how affects fit chi^2
#disc = 'as209_test3'
#disc = 'sz114'
#disc = 'sz71'
#disc = 'sz98'
#disc = 'citau'
#disc = 'citau_test'
#disc = 'synth_gauss'
#disc = 'synth_gauss_reverse'
#disc = 'synth_gauss_ring'
#disc = 'synth_gauss_ring_moreruns'
#disc = 'synth_gauss_ring_fwhm_pt1_test'
#disc = 'synth_gauss_ring_fwhm_pt05'
#disc = 'synth_gauss_ring_fwhm_pt025'
#disc = 'synth_gauss_ring_fwhm_pt025_small_rout
#disc = 'synth_gauss_ring_fwhm_pt0125'
#disc = 'synth_richard_profile_pt1_smallest'
#disc = 'synth_dsharp'

#disc = 'double_gauss'
#disc = 'smooth_sharp_test'

#disc = 'synth_gauss_ring_test'

#disc = 'asym_gauss_ring_test'

#disc = 'synth_richard_profile_for_clean_comparison13' # compressed disc
#disc = 'synth_richard_profile_original' # disc out to 3"
#disc = 'synth_richard_profile_original_5e10'
#disc = 'synth_gauss_rev3'
#disc = 'asym_gauss_ring_rev3'
#disc = 'double_gauss_rev3'
#disc = 'gauss_ring_pt0125_rev3'
#disc = 'gauss_ring_pt0125_briggs_negative2_rev3'
#disc = 'gauss_ring_pt025_rev3'
#disc = 'dsharp_rev3'
#disc = 'dsharp_5e10_rev3'
#disc = 'dsharp_C437_rev3'
#disc = 'dsharp_5e10_C437_rev3'
#disc = 'dsharp_5e10_C437_40minuteintegration_rev3'
#disc = 'citau_realdata_rev3'
#disc = 'citau_gr_band7'
#disc = 'citau_rev3_largerRout'
#disc = 'citau_rev3_C437'
#disc = 'citau_rev3_C437_sharpouteredge'
#disc = 'narrow_ring_on_broad_envelope_rev3'

#disc = 'as209'
#disc = 'citau'
#disc = 'rulup'

"""
if disc == 'rulup':
    #inc, pa, dra, ddec, wle = [-0.0016508156666454792,  -8.774298583226126e-05, 0.014217070840063635 / rad_to_arcsec / deg_to_rad, -0.0853152383800024 / rad_to_arcsec / deg_to_rad, 0.0012566903032206866]#.0012552301255230125]#0.00130344546957]
    inc, pa, dra, ddec, wle = [18.8, 121., 17.1e-3 / rad_to_arcsec / deg_to_rad, -88.1e-3 / rad_to_arcsec / deg_to_rad, 0.0012566903032206866]
    Rmax = .65
    uvtable_filename = ['RULup_continuum_nopnt_nofl_NTlBhwg.npz']

if disc == 'narrow_ring_on_broad_envelope_rev3':
    inc, pa, dra, ddec, wle = [0., 0., 0., 0., 0.00130344546957]
    Rmax = 1.
    uvtable_filename = ['narrow_ring_on_broad_envelope_rev3.alma.cycle6.6.noisy.ms_uvtable.txt']

    def model(r):
        normalization = 1e10

        def gauss(x, a, b, c, d):  # a = peak mag., b = x-val at peak, c ~ bell width, d = y-offset
            return a * np.exp(-(x - b) ** 2 / (2. * c ** 2)) + d

        return gauss(r, .2 * normalization, 0 / rad_to_arcsec, .7 / rad_to_arcsec, 0) \
               + gauss(r, 1. * normalization, .5 / rad_to_arcsec, .025 / rad_to_arcsec, 0)


if disc == 'double_gauss':
    inc, pa, dra, ddec, wle = [0., 0., 0., 0., 0.00130344546957]
    Rmax = 1.5
    uvtable_filename = ['synth_double_gauss_inc0_B6_C436.alma.cycle6.6.noisy.ms_uvtable.txt']

    def model(r):
        from scipy.interpolate import interp1d
        rs, bs = np.genfromtxt(pwd + '/uvtables/' + 'double_gauss_narrow_broad_brightness_profile.txt', skip_header=1).T
        interp_f = interp1d(rs, bs, kind='linear', bounds_error=False, fill_value=(bs[0], bs[-1]))
        return interp_f(r)

if disc == 'double_gauss_rev3':
    inc, pa, dra, ddec, wle = [0., 0., 0., 0., 0.00130344546957]
    Rmax = 1.2
    uvtable_filename = ['double_gauss_rev3.alma.cycle6.6.noisy.ms_uvtable.txt']

    def model(r):
        normalization = 1e10

        def gauss(x, a, b, c, d):  # a = peak mag., b = x-val at peak, c ~ bell width, d = y-offset
            return a * np.exp(-(x - b) ** 2 / (2. * c ** 2)) + d

        def sigmoid(x, a, c1, c2, d, middle):
            half1 = gauss(x, a, middle, c1, d) * ((x <= middle))
            half2 = gauss(x, a, middle, c2, d) * ((x > middle))
            return half1 + half2
            # return a * np.exp(-(x - middle)**2 / (2. * c1**2)) + d

        return gauss(r, normalization, 0 / rad_to_arcsec, .2 / rad_to_arcsec, 0) + \
               sigmoid(r, normalization, .1 / rad_to_arcsec, .2 / rad_to_arcsec, 0., .6 / rad_to_arcsec)

if disc == 'smooth_sharp_test':
    inc, pa, dra, ddec, wle = [0., 0., 0., 0., 0.00130344546957]
    Rmax = 5.
    uvtable_filename = ['synth_smooth_sharp_edges_test_inc0_B6_C436.alma.cycle6.6.noisy.ms_uvtable.txt']
    '''
    def model(r):
        from scipy.interpolate import interp1d
        rs, bs = np.genfromtxt(pwd + '/uvtables/' + 'smooth_sharp_edges_test_brightness_profile.txt', skip_header=1).T
        interp_f = interp1d(rs, bs, kind='linear', bounds_error=False, fill_value=(bs[0], bs[-1]))
        return interp_f(r)
    '''

    def gauss(x, a, b, c, d):  # a = peak mag., b = x-val at peak, c ~ bell width, d = y-offset
        return a * np.exp(-(x - b) ** 2 / (2. * c ** 2)) + d


    def double_gauss(x, a1, a2, b1, db, c1, c2, offset):
        return gauss(x, a1, b1, c1, offset / 2.) + \
               gauss(x, a2, b1 + abs(db), c2, offset / 2.)


    def tophat(x, amp, middle, width):
        return amp * ((x >= middle - width / 2.) & (x <= middle + width / 2.))


    def sigmoid(x, a, c1, c2, d, middle):
        half1 = gauss(x, a, middle, c1, d) * ((x <= middle))
        half2 = gauss(x, a, middle, c2, d) * ((x > middle))
        return half1 + half2


    def edges_test(x, a, b, c, d, box_middle, box_width, c1, c2, sigmoid_middle):
        return 5e10 - .9 * gauss(x, a, b, c, d) - \
               .9 * tophat(x, a, box_middle, box_width) - \
               .9 * sigmoid(x, a, c1, c2, d, sigmoid_middle)

    def model(r):
        normalization = 5e10
        return edges_test(r, normalization, .75 / rad_to_arcsec, .2 / rad_to_arcsec, 0., 1.9 / rad_to_arcsec,
                          .5 / rad_to_arcsec, .15 / rad_to_arcsec, .5 / rad_to_arcsec, 3. / rad_to_arcsec) # smooth_sharp_edges_test

if disc == 'synth_dsharp' or disc == 'dsharp_rev3' or disc == 'dsharp_5e10_rev3' or disc == 'dsharp_C437_rev3' \
        or disc == 'dsharp_5e10_C437_rev3' or disc == 'dsharp_5e10_C437_40minuteintegration_rev3':
    inc, pa, dra, ddec, wle = [0., 0., 0., 0., 0.00130344546957]
    Rmax = 1.5
    uvtable_filename = ['synth_dsharp_profile_inc0_B6_C436.alma.cycle6.6.noisy.ms_uvtable.txt']

    if disc == 'dsharp_rev3':
        Rmax = 1.35
        uvtable_filename = ['dsharp_rev3.alma.cycle6.6.noisy.ms_uvtable.txt']

    if disc == 'dsharp_5e10_rev3':
        Rmax = 1.35
        uvtable_filename = ['dsharp_5e10_rev3.alma.cycle6.6.noisy.ms_uvtable.txt']

    if disc == 'dsharp_C437_rev3':
        Rmax = 1.35
        uvtable_filename = ['dsharp_C437_rev3.alma.cycle6.7.noisy.ms_uvtable.txt']

    if disc == 'dsharp_5e10_C437_rev3':
        Rmax = 1.35
        uvtable_filename = ['dsharp_5e10_C437_rev3.alma.cycle6.7.noisy.ms_uvtable.txt']

    if disc == 'dsharp_5e10_C437_40minuteintegration_rev3':
        Rmax = 1.35
        uvtable_filename = ['dsharp_5e10_C437_40minuteintegration_rev3.alma.cycle6.7.noisy.ms_uvtable.txt']

    def model(r):
        normalization = 5e10
        if disc == 'dsharp_rev3' or disc == 'dsharp_C437_rev3': normalization = 1e10
        from scipy.interpolate import interp1d
        rs, bs = np.genfromtxt('/Users/morgan/gdrive_jmjenn/recipes/Frankenstein/uvtables/dsharp_as209.csv', delimiter=',').T
        rs /= 121.
        bs = bs / max(bs) * normalization
        interp_f = interp1d(rs / rad_to_arcsec, bs, kind='linear', bounds_error=False, fill_value=(bs[0], bs[-1]))



if disc == 'synth_gauss_ring' or disc == 'synth_gauss_ring_moreruns' or disc == 'synth_gauss_ring_fwhm_pt1_test' \
        or disc == 'synth_gauss_ring_fwhm_pt05' or disc == 'synth_gauss_ring_fwhm_pt025' or disc == 'synth_gauss_ring_fwhm_pt025_small_rout' \
        or disc == 'synth_gauss_ring_fwhm_pt0125' or disc == 'synth_gauss_ring_test' or disc == 'gauss_ring_pt0125_rev3' or disc == 'gauss_ring_pt025_rev3' \
        or disc == 'gauss_ring_pt0125_briggs_negative2_rev3':
    inc, pa, dra, ddec, wle = [0., 0., 0., 0., 0.00130344546957]

    Rmax = 2.
    if disc == 'synth_gauss_ring_fwhm_pt1_test': Rmax = 1.
    if disc == 'synth_gauss_ring_fwhm_pt05': Rmax = 1.
    if disc == 'synth_gauss_ring_fwhm_pt025': Rmax = 1.
    if disc == 'synth_gauss_ring_fwhm_pt025_small_rout': Rmax = .7
    if disc == 'synth_gauss_ring_fwhm_pt0125': Rmax = 1.
    if disc == 'synth_gauss_ring_test': Rmax = 3.

    uvtable_filename = ['synth_gauss_ring_inc0_B6_C436.alma.cycle6.6.noisy.ms_uvtable.txt']
    if disc == 'synth_gauss_ring_fwhm_pt1_test': uvtable_filename = ['synth_gauss_ring_fwhm_pt1.alma.cycle6.6.noisy.ms_uvtable.txt']
    if disc == 'synth_gauss_ring_fwhm_pt05': uvtable_filename = ['synth_gauss_ring_fwhm_pt05.alma.cycle6.6.noisy.ms_uvtable.txt']
    if disc == 'synth_gauss_ring_fwhm_pt025' or disc == 'synth_gauss_ring_fwhm_pt025_small_rout': uvtable_filename = ['synth_gauss_ring_fwhm_pt025.alma.cycle6.6.noisy.ms_uvtable.txt']
    if disc == 'synth_gauss_ring_fwhm_pt0125': uvtable_filename = ['synth_gauss_ring_fwhm_pt0125.alma.cycle6.6.noisy.ms_uvtable.txt']
    if disc == 'synth_gauss_ring_test': uvtable_filename == ['synth_gauss_ring_inc0_B6_C436_test.alma.cycle6.6.noisy.ms_uvtable.txt']

    def gauss(x, a, b, c, d):  # a = peak mag., b = x-val at peak, c ~ bell width, d = y-offset
        return a * np.exp(-(x - b) ** 2 / (2. * c ** 2)) + d

    def model(r):
        #x = r * rad_to_arcsec
        normalization = 5e10
        #R = np.linspace(Rin, Rin + dR * nR, nR, endpoint=False)
        return gauss(r, normalization, 1 / rad_to_arcsec, .2 / rad_to_arcsec, 0)

    if disc == 'synth_gauss_ring_fwhm_pt1_test':
        def model(r):
            # x = r * rad_to_arcsec
            normalization = 5e10
            # R = np.linspace(Rin, Rin + dR * nR, nR, endpoint=False)
            return gauss(r, normalization, .5 / rad_to_arcsec, .1 / rad_to_arcsec, 0)

    if disc == 'synth_gauss_ring_fwhm_pt05':
        def model(r):
            normalization = 5e10
            return gauss(r, normalization, .5 / rad_to_arcsec, .05 / rad_to_arcsec, 0)

    if disc == 'synth_gauss_ring_fwhm_pt025' or disc == 'synth_gauss_ring_fwhm_pt025_small_rout':
        def model(r):
            normalization = 5e10
            return gauss(r, normalization, .5 / rad_to_arcsec, .025 / rad_to_arcsec, 0)

    if disc == 'synth_gauss_ring_fwhm_pt0125':
        def model(r):
            normalization = 5e10
            return gauss(r, normalization, .5 / rad_to_arcsec, .0125 / rad_to_arcsec, 0)

    if disc == 'synth_gauss_ring_test':
        def model(r):
            normalization = 5e10
            return gauss(r, normalization, 1. / rad_to_arcsec, .2 / rad_to_arcsec, 0)

    if disc == 'gauss_ring_pt0125_rev3':
        uvtable_filename = ['gauss_ring_pt0125_rev3.alma.cycle6.6.noisy.ms_uvtable.txt']
        Rmax = 1.
        def model(r):
            normalization = 1e10
            return gauss(r, normalization, .5 / rad_to_arcsec, .0125 / rad_to_arcsec, 0)

    if disc == 'gauss_ring_pt0125_briggs_negative2_rev3':
        uvtable_filename = ['gauss_ring_pt0125_briggs_negative2_rev3.alma.cycle6.6.noisy.ms_uvtable.txt']
        Rmax = 1.

        def model(r):
            normalization = 1e10
            return gauss(r, normalization, .5 / rad_to_arcsec, .0125 / rad_to_arcsec, 0)

    if disc == 'gauss_ring_pt025_rev3':
        uvtable_filename = ['gauss_ring_pt025_rev3.alma.cycle6.6.noisy.ms_uvtable.txt']
        Rmax = 1.
        def model(r):
            normalization = 1e10
            return gauss(r, normalization, .5 / rad_to_arcsec, .025 / rad_to_arcsec, 0)

if disc == 'asym_gauss_ring_test' or disc == 'asym_gauss_ring_rev3':
    inc, pa, dra, ddec, wle = [0., 0., 0., 0., 0.00130344546957]
    Rmax = 1.2
    #uvtable_filename = ['asym_gauss_ring11.alma.cycle6.6.noisy.ms_uvtable.txt']
    #uvtable_filename = ['asym_gauss_ring2.alma.cycle6.6.noisy.ms_uvtable.txt']
    #uvtable_filename = ['asym_gauss_ring3.alma.cycle6.6.noisy.ms_uvtable.txt']
    #uvtable_filename = ['asym_gauss_ring4.alma.cycle6.6.noisy.ms_uvtable.txt']
    #uvtable_filename = ['asym_gauss_ring5.alma.cycle6.6.noisy.ms_uvtable.txt']
    #uvtable_filename = ['asym_gauss_ring7.alma.cycle6.6.noisy.ms_uvtable.txt']
    #uvtable_filename = ['asym_gauss_ring8.alma.cycle6.6.noisy.ms_uvtable.txt']
    #uvtable_filename = ['asym_gauss_ring9.alma.cycle6.6.noisy.ms_uvtable.txt']
    #uvtable_filename = ['asym_gauss_ring10.alma.cycle6.6.noisy.ms_uvtable.txt']
    uvtable_filename = ['asym_gauss_ring_rev3.alma.cycle6.6.noisy.ms_uvtable.txt']

    def gauss(x, a, b, c, d):  # a = peak mag., b = x-val at peak, c ~ bell width, d = y-offset
        return a * np.exp(-(x - b) ** 2 / (2. * c ** 2)) + d

    def sigmoid(x, a, c1, c2, d, middle):
        half1 = gauss(x, a, middle, c1, d) * ((x <= middle))
        half2 = gauss(x, a, middle, c2, d) * ((x > middle))
        return half1 + half2

    def model(r):
        normalization = 5e10
        if disc == 'asym_gauss_ring_rev3':
            normalization = 1e10
            return sigmoid(r, normalization, .1 / rad_to_arcsec, .2 / rad_to_arcsec, 0., .6 / rad_to_arcsec)
        return sigmoid(r, normalization, .2 / rad_to_arcsec, .1 / rad_to_arcsec, 0., .7 / rad_to_arcsec)
        #return gauss(r, normalization, .5 / rad_to_arcsec, .1 / rad_to_arcsec, 0.)


if disc == 'synth_gauss' or disc == 'synth_gauss_reverse' or disc == 'synth_gauss_rev3':
    inc, pa, dra, ddec, wle = [0., 0., 0., 0., 0.00130344546957]
    Rmax = .7
    #Rmax = 1.4
    uvtable_filename = ['synth_gauss_inc0_B6_C436.alma.cycle6.6.noisy.ms_uvtable.txt']
    if disc == 'synth_gauss_rev3':
        uvtable_filename = ['synth_gauss_rev3.alma.cycle6.6.noisy.ms_uvtable.txt']

    def gauss(x, a, b, c, d):  # a = peak mag., b = x-val at peak, c ~ bell width, d = y-offset
        return a * np.exp(-(x - b) ** 2 / (2. * c ** 2)) + d

    def model(r):
        #x = r * rad_to_arcsec
        normalization = 5e10
        if disc == 'synth_gauss_rev3':
            normalization = 1e10
        #R = np.linspace(Rin, Rin + dR * nR, nR, endpoint=False)
        return gauss(r, normalization, 0 / rad_to_arcsec, .2 / rad_to_arcsec, 0)




if disc == 'citau_realdata_rev3' or disc == 'citau_test' or disc == 'citau_gr_band7': # real, hi-res obs
    inc, pa, dra, ddec, wle, wcorr = [47.19418051074036, 14.156835217251384, -0.33340551644789906 / rad_to_arcsec / deg_to_rad,
                                      0.09462042532030927 / rad_to_arcsec / deg_to_rad, 0.00130347953005, 1.]
    if disc == 'citau_gr_band7':
        inc, pa, dra, ddec, wle, wcorr = [47.19418051074036, 14.156835217251384, -0.33340551644789906 / rad_to_arcsec / deg_to_rad,
                                          0.09462042532030927 / rad_to_arcsec / deg_to_rad, 0.0008695652173913044, 1.]

    Rmax = 3.

    uvtable_filename = ['uvtable_new_nofl_wcorr_30s.txt']

    if disc == 'citau_gr_band7':
        uvtable_filename = ['CI_Tau_cont_fourspws.txt']

if disc == 'citau' or disc == 'citau_rev3_largerRout' or disc == 'citau_rev3_C437' or disc == 'citau_rev3_C437_sharpouteredge': # synthetic dataset set up as as209-like mid-res obs
    inc, pa, dra, ddec, wle = [0., 0., 0., 0., 0.00130344546957]
    Rmax = 1.25
    uvtable_filename = ['citau.alma.cycle6.6.noisy.ms_uvtable.txt']

    if disc == 'citau_rev3_C437':
        uvtable_filename = ['citau_rev3_C437.alma.cycle6.7.noisy.ms_uvtable.txt']

    if disc == 'citau_rev3_C437_sharpouteredge':
        uvtable_filename = ['citau_rev3_C437_sharpouteredge.alma.cycle6.7.noisy.ms_uvtable.txt']

    if disc == 'citau_rev3_largerRout':
        Rmax = 1.35
        uvtable_filename = ['citau_rev3_largerRout.alma.cycle6.6.noisy.ms_uvtable.txt']

    def model(r):
        from scipy.interpolate import interp1d
        normalization = 6e10
        rs, bs = np.genfromtxt('/Users/morgan/gdrive_jmjenn/recipes/Frankenstein/uvtables/citau_frompaper_digitized.csv', delimiter=',').T
        rs /= 160.
        if disc == 'citau_rev3_C437_sharpouteredge':
            rs = np.append(rs, rs[-1] + np.diff(rs)[-1])
            bs = np.append(bs, bs[-1] * 1e-2)
        bs = bs / max(bs) * normalization
        interp_f = interp1d(rs / rad_to_arcsec, bs, kind='linear', bounds_error=False, fill_value=(bs[0], bs[-1]))
        return interp_f(r)

if disc == 'sz114':
    inc, pa, dra, ddec, wle, wcorr = [15.84, 148.73,  0.02 / rad_to_arcsec / deg_to_rad, -0.03 / rad_to_arcsec / deg_to_rad, 0.00116548355051, 1.]
    Rmax = .75

    uvtable_filename = ['sz114_cont.txt']


if disc == 'sz71':
    inc, pa, dra, ddec, wle, wcorr = [37.51, 40.82, -0.18 / rad_to_arcsec / deg_to_rad, -0.13 / rad_to_arcsec / deg_to_rad, 0.00116548355051, 1.]
    Rmax = 2.

    uvtable_filename = ['sz71_cont.txt']


if disc == 'sz98':
    inc, pa, dra, ddec, wle, wcorr = [47.1, 111.58, -0.02 / rad_to_arcsec / deg_to_rad, 0.01 / rad_to_arcsec / deg_to_rad, 0.00116548355051, 1.]
    Rmax = 2.

    uvtable_filename = ['sz98_cont.txt']


if disc == 'mock1' or disc == 'mock1_test' or disc == 'mock1_uvtable_mod_fittedi30' or disc == 'mock1_highi':
    inc, pa, dra, ddec, wle = [30., 0., 0., 0., 0.00130344546957]#1.]#.0166]
    Rmax = .7

    uvtable_filename = ['mock1_uvtable.txt']

    def model(r):
        x = r * rad_to_arcsec
        env = 1. * rad_to_arcsec**2
        #env = 5e10 # temporary
        env /= 5

        gap1 = 0.1 * ((x > .1) & (x < .2))
        gap2 = 0.92 * ((x > .2) & (x < .3))
        gap3 = 0.85 * ((x > .3) & (x < .4))
        gap4 = 1 / 3. * ((x > .4) & (x < .5))
        gap5 = 0.8 * ((x > .5) & (x < .6))
        gap6 = 0.45 * ((x > .6) & (x < .65))
        gap7 = 1. * ((x > .65) & (x <= .7))

        return env * (1 - gap1 - gap2 - gap3 - gap4 - gap5 - gap6 - gap7)

    if disc == 'mock1_highi':
        inc = 85.
        #wcorr = .0217 # temporary. wrong?

if disc == 'mock2':
    inc, pa, dra, ddec, wle, wcorr = [0., 0., 0., 0., 0.00130344546957, 0.008634316691939398]
    Rmax = 1.

    uvtable_filename = [
        'mock2_inc0_B6_C436.alma.cycle6.6.noisy.ms_uvtable.txt',
        'mock2_inc0_B6_C437.alma.cycle6.7.noisy.ms_uvtable.txt',
        'mock2_inc0_B6_C438.alma.cycle6.8.noisy.ms_uvtable.txt',
        'mock2_inc0_B6_C439.alma.cycle6.9.noisy.ms_uvtable.txt',
    ]
    # uvtable_filename = [uvtable_filename[1]]

    def model(r):
        x = r * rad_to_arcsec
        env = 1. * rad_to_arcsec ** 2
        env = 5e10  # temporary

        gap1 = 0.1 * ((x > .05) & (x < .1))
        gap2 = 0.92 * ((x > .1) & (x < .15))
        gap3 = 0.85 * ((x > .15) & (x < .2))
        gap4 = 0.33 * ((x > .2) & (x < .3))
        gap5 = 0.73 * ((x > .3) & (x < .35))
        gap6 = 0.88 * ((x > .35) & (x < .4))
        gap7 = 0.55 * ((x > .4) & (x < .475))
        gap8 = 1. * ((x > .475))

        return env * (1 - gap1 - gap2 - gap3 - gap4 - gap5 - gap6 - gap7 - gap8)
"""
