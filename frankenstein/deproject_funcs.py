import numpy as np
from scipy.optimize import minimize
import os
pwd = os.getcwd()
import copy

from uvplot import UVTable
from uvplot import COLUMNS_V0, COLUMNS_V2

from params import savedir, disc # TODO: clean up. if funcs take disc and savedir as input (they should probably), redundant to import here
import data_funcs
import plot_funcs

from constants import c

def adjust_disc(uv, dra, ddec, inc, pa, wle, use_freqs, disc, savedir, known_geometry, input_kl=False, to_plot=False):
    '''Apply phase shift by dra, ddec to re,im; rotate u,v by pa; deproject u,v by inc; normalize u,v by wle'''
    print('  Fitting and correcting data for disc geometry')

    if use_freqs:
        uv.u /= c / uv.freqs
        uv.v /= c / uv.freqs
    else:
        uv.u /= wle
        uv.v /= wle

    if input_kl:
        uv.u /= 1e3
        uv.v /= 1e3

    uv_pre = copy.deepcopy(uv)

    if not known_geometry:
        from constants import rad_to_arcsec
        u, v, real, imag, weights = np.copy(uv.u), np.copy(uv.v), np.copy(uv.re), np.copy(uv.im), np.copy(uv.weights) # TODO: fix this shit
        u /= rad_to_arcsec
        v /= rad_to_arcsec
        def deproject(u, v, inc, pa):
            cos_t = np.cos(pa)
            sin_t = np.sin(pa)

            up = u * cos_t - v * sin_t
            vp = u * sin_t + v * cos_t
            #   De-project
            up *= np.cos(inc)

            return up, vp

        def apply_phase_shift(u, v, vis, dRA, dDec):
            dRA *= 2. * np.pi
            dDec *= 2. * np.pi

            phi = u * dRA + v * dDec

            return vis * (np.cos(phi) + 1j * np.sin(phi))

        def gauss(params):
            dRA, dDec, inc, pa, norm, scal = params

            vis = apply_phase_shift(u, v, real + 1j * imag, dRA, dDec)

            up, vp = deproject(u, v, inc, pa)

            # Evaluate the gaussian:
            gaus = np.exp(- 0.5 * (up ** 2 + vp ** 2) / scal ** 2)

            # Evaluate at the Chi2
            chi2 = weights * np.abs(norm * gaus - vis) ** 2
            return chi2.sum() / (2 * len(weights))

        res = minimize(gauss, [0., 0., .1, .1, 1., 1.])

        # dRA, dDec are in arcsec
        # inc, pa are in radians
        dra, ddec, inc, pa, norm, scal = res.x
        dra /= rad_to_arcsec # TODO: redo all this, don't switch b/t units internally
        ddec /= rad_to_arcsec
        '''
        print('dra, ddec (arcsec):', Dra, Ddec)
        print('inclination, position angle (deg)', Inc * 180 / np.pi, Pa * 180 / np.pi)
        print('Gaussian normalization, scale factor', norm, scal)
        print('Chi2:', res.fun)
        '''

    uv.apply_phase(dra, ddec) # using uvplot
    uv.deproject(np.abs(inc), pa) # using uvplot
    if to_plot: plot_funcs.plot_adjust_disc(uv_pre, uv, wle, disc, savedir, dra, ddec, inc, pa)

    return uv


def fit_disc_geometry(uvtable_filename, dra, ddec, inc, pa, wle, use_freqs, disc, savedir, known_geometry, input_kl, cutuv, maxuv, set_im_to_0, snr_plot_binwidth, known_profile, known_weights, to_plot):
    uv, baselines = data_funcs.load_obs(uvtable_filename, known_profile)

    Rmax = 1.3 * 0.574 * wle / np.min(baselines) # maximum recoverable scale # TODO: just set Rmax based on max baseline conversion (+ some buffer)
    # + some buffer
    resmax = 0.574 * wle / np.max(baselines) # highest achievable data resolution
    ncolpts = np.int(Rmax / resmax) # number of collocation points used in the fit

    uv = adjust_disc(uv, dra, ddec, inc, pa, wle, use_freqs, disc, savedir, known_geometry, input_kl, to_plot)

    if cutuv: uv = uv.uvcut(maxuv)
        #uvcut = uv.uvdist>maxuv
        #uv = UVTable(uvtable=[a[uvcut] for a in [uv.u, uv.v, uv.V, uv.weights, uv.freqs, uv.spws]],
        #columns=COLUMNS_V2)

    if set_im_to_0: uv.im *= 0.
    # TODO: is chi2 including Im(v)?

    baselines = np.hypot(uv.u, uv.v)
    #if known_profile:
    if known_weights: wcorr = 1.
    else: wcorr = data_funcs.find_wcorr(baselines, uv) # TODO: reintegrate this
    uv.weights /= wcorr**2
    if known_profile:
        #np.savetxt(pwd + '/../ALMAsim/' + disc + '/' + os.path.splitext(uvtable_filename[-1])[0] + '_corrected'
        #       + os.path.splitext(uvtable_filename[-1])[1], np.stack([uv.u, uv.v, uv.re, uv.im, uv.weights], axis=-1))
        np.savetxt(pwd + '/uvtables/' + os.path.splitext(uvtable_filename[-1])[0] + '_corrected'
               + os.path.splitext(uvtable_filename[-1])[1], np.stack([uv.u, uv.v, uv.re, uv.im, uv.weights], axis=-1))
    else: np.savetxt(pwd + '/uvtables/' + os.path.splitext(uvtable_filename[-1])[0] + '_corrected'
               + os.path.splitext(uvtable_filename[-1])[1], np.stack([uv.u, uv.v, uv.re, uv.im, uv.weights], axis=-1)) # TODO: save as uv.weights * wcorr ** 2?
    # TODO: add option to save uv.freqs?
    #if to_plot: plot_funcs.plot_snr(uv, snr_plot_binwidth, disc, savedir)

    '''
    plt.figure()
    plt.loglog(baselines, uv.weights, '.')
    plt.savefig(savedir + 'weights.png')
    plt.close()
    '''

    return uv, baselines, wcorr
