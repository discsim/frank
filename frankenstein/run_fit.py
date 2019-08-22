import numpy as np

from params import *
from fitter import GPHankelFitter
import data_funcs
import deproject_funcs
import plot_funcs
from constants import rad_to_arcsec

from pathlib import Path

Path(savedir).mkdir(exist_ok=True) # if the save directory doesn't exist, create it

print('\nFitting %s:'%disc)

Rmax, dra, ddec, pa, inc = data_funcs.convert_units(Rmax, dra, ddec, pa, inc)

#if known_geometry:
#    uv, baselines = data_funcs.load_obs([os.path.splitext(uvtable_filename[-1])[0] + '_corrected' + os.path.splitext(uvtable_filename[-1])[1]], known_profile)#, add_noise, noise)
#else:
uv, baselines, wcorr_guess = deproject_funcs.fit_disc_geometry(uvtable_filename, dra, ddec, inc, pa, wle, use_freqs, disc, savedir, known_geometry, input_kl, cutuv, maxuv, set_im_to_0, snr_plot_binwidth, known_profile, known_weights, to_plot=True)

#l80 = np.sort(baselines)[round(.8*len(baselines))]
#alma_res = .574 * wle / l80 * rad_to_arcsec
#print('80th percentile of baselines',l80,r'\lambda. min',min(baselines), r'\lambda. max',max(baselines), r'\lambda. approx spatial res.',alma_res,'arcsec')

'''
if add_noise:
    sortidx = np.argsort(baselines)
    #to_add_noise = sortidx[round(.8 * len(baselines)):]
    to_add_noise = sortidx[:round(.2 * len(baselines))]
    print('mean(uv.re)',np.mean(uv.re[to_add_noise]))
    random_draws = np.random.normal(size=len(uv.u[to_add_noise]))
    noise = noise_ratio * uv.re[to_add_noise] * random_draws
    print('adding noise of mean amplitude',np.mean(noise))
    mean_ratio = np.mean(noise / uv.re[to_add_noise])
    print('mean ratio',mean_ratio)
    uv.re[to_add_noise] += noise
'''

counter_gp = 0
nfits = 1
''' # run fits in loop:
nfits = len(nbins) * len(alphas) * len(smooth_strength)
for nbins in nbins_list:
    for alpha in alphas:
        for smooth_strength in smooth_strengths:
'''
print('  Running fit with Rmax=%.1f", %s collocation points (nominal resolution %.4f"), alpha=%s, smooth_strength=%s' % (Rmax * rad_to_arcsec,nbins,Rmax * rad_to_arcsec / nbins, alpha,smooth_strength))
GPHF = GPHankelFitter(Rmax, nbins, alpha=alpha, enforce_positive=enforce_positivity)#, q=0)
mu, cov_renorm, uv_pts_toplot, pi_toplot, mu_toplot, alphas_toplot, pointwise_sc_toplot, stop_criterion_toplot, smooth_iterations = \
    GPHF.fit(baselines, uv.re, smooth_strength, uv.weights)

#r0, mu0 = np.genfromtxt(savedir+'fit_mu0.txt').T
#print('mean',np.mean(mu[:-1]/mu0))

cut = (np.abs(GPHF.Rc - Rmax)).argmin()  # TODO: is the fit not set to 0 outside Rmax? TODO: rewrite this for discs w/o known profiles.
# for plotting fit only out to given Rmax. equivalent to GPHF.Rc[:len(GPHF.Rc) / <factor by which multiply Rmax>]

if save_fit_values:
    fit_savefile = savedir + 'fit.txt'
    print('  Saving fit values to', '/'.join(Path(fit_savefile).parts[-3:]))
    np.savetxt(fit_savefile, np.array([GPHF.Rc[:cut] * rad_to_arcsec, mu[:cut]]).T, header='r [arcsec]\tI [Jy/sr]')

if plot_results:
    plot_savefile = savedir + 'fit.png'
    print('  Plotting fit, saving figure to', '/'.join(Path(plot_savefile).parts[-3:]))
    plot_funcs.plot_fit(disc, Rmax, baselines, uv, model, GPHF, mu, nbins, alpha, smooth_strength, uvbin_size, uv_pts_toplot, pi_toplot, mu_toplot, alphas_toplot, pointwise_sc_toplot, stop_criterion_toplot, enforce_positivity, smooth_iterations, cut, counter_gp, known_profile, image_extracted_profile, add_noise, nfits, cmaps, snr_plot_binwidth, plot_savefile, savedir)
if ms_figs:
    plot_savefile = savedir + 'fit.png'
    plot_msfigs_savefile = savedir + 'ms_draft.png'
    plot_funcs.plot_msfigs(plot_msfigs_savefile, disc, Rmax, baselines, uv, model, GPHF, mu, nbins, alpha, smooth_strength, uvbin_size, uv_pts_toplot, pi_toplot, enforce_positivity, smooth_iterations, cut, counter_gp, known_profile, image_extracted_profile, add_noise, nfits, cmaps, snr_plot_binwidth, plot_savefile, savedir)
    #plot_funcs.plot_msfigs_2by2(plot_msfigs_savefile, disc, Rmax, baselines, uv, model, GPHF, mu, nbins, alpha, smooth_strength, chi2s, uvbin_size, uv_pts_toplot, pi_toplot, enforce_positivity, smooth_iterations, cut, counter_gp, known_profile, image_extracted_profile, add_noise, nfits, cmaps, snr_plot_binwidth, plot_savefile, savedir)
#counter_gp += 1

if ms_f1:
    plot_savefile = savedir + 'fit.png'
    plot_msfigs_savefile = savedir + 'ms_f1_draft.png'
    plot_funcs.ms_f1(plot_msfigs_savefile, disc, Rmax, baselines, uv, model, GPHF, mu, nbins, alpha, smooth_strength, uvbin_size, uvbin_size2, uv_pts_toplot, pi_toplot, enforce_positivity, smooth_iterations, cut, counter_gp, known_profile, image_extracted_profile, add_noise, nfits, cmaps, snr_plot_binwidth, snr_plot_binwidth2, plot_savefile, savedir)

if ms_f2:
    plot_savefile = savedir + 'fit.png'
    plot_msfigs_savefile = savedir + 'ms_f2_draft.png'
    plot_funcs.ms_f2(plot_msfigs_savefile, disc, Rmax, baselines, uv, model, GPHF, mu, nbins, alpha, smooth_strength, uvbin_size, uvbin_size2, uv_pts_toplot, pi_toplot, enforce_positivity, smooth_iterations, cut, counter_gp, known_profile, image_extracted_profile, add_noise, nfits, cmaps, snr_plot_binwidth, snr_plot_binwidth2, plot_savefile, savedir)

if ms_f3:
    plot_savefile = savedir + 'fit.png'
    plot_msfigs_savefile = savedir + 'ms_f3_draft.png'
    plot_funcs.ms_f3(plot_msfigs_savefile, disc, Rmax, baselines, uv, model, GPHF, mu, nbins, alpha, smooth_strength, uvbin_size, uvbin_size2, uv_pts_toplot, pi_toplot, enforce_positivity, smooth_iterations, cut, counter_gp, known_profile, image_extracted_profile, add_noise, nfits, cmaps, snr_plot_binwidth, snr_plot_binwidth2, plot_savefile, savedir)

if ms_f4:
    plot_savefile = savedir + 'fit.png'
    plot_msfigs_savefile = savedir + 'ms_f4_draft.png'
    plot_msfigs_savefile2 = savedir + 'ms_f4_draft_pt2.png'
    plot_funcs.ms_f4(plot_msfigs_savefile, plot_msfigs_savefile2, disc, Rmax, baselines, uv, model, GPHF, mu, nbins, alpha, smooth_strength, uvbin_size, uvbin_size2, uv_pts_toplot, pi_toplot, enforce_positivity, smooth_iterations, cut, counter_gp, known_profile, image_extracted_profile, add_noise, nfits, cmaps, snr_plot_binwidth, snr_plot_binwidth2, plot_savefile, savedir)

if ms_f5:
    plot_savefile = savedir + 'fit.png'
    plot_msfigs_savefile = savedir + 'ms_f5_draft.png'
    plot_msfigs_savefile2 = savedir + 'ms_f5_draft_pt2.png'
    plot_funcs.ms_f5(plot_msfigs_savefile, plot_msfigs_savefile2, disc, Rmax, baselines, uv, model, GPHF, mu, nbins, alpha, smooth_strength, uvbin_size, uvbin_size2, uv_pts_toplot, pi_toplot, enforce_positivity, smooth_iterations, cut, counter_gp, known_profile, image_extracted_profile, add_noise, nfits, cmaps, snr_plot_binwidth, snr_plot_binwidth2, plot_savefile, savedir)    

print("IT'S ALIVE!\n")
