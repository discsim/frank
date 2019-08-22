import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
import os
pwd = os.getcwd()

from constants import rad_to_arcsec, deg_to_rad

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def plot_log_abs(x, y, ax, **kwargs):
    '''Plot log |y| vs log x, w/ negative ys shown dashed'''
    c = ax.loglog(x, y, **kwargs)[0].get_color()
    kwargs = dict(kwargs)
    kwargs['c'] = c
    kwargs['label'] = None
    kwargs['ls'] = '--'
    ax.loglog(x, -y, **kwargs)


def plot_adjust_disc(uv_pre, uv_post, wle, disc, savedir, Dra, Ddec, Inc, Pa):
    import matplotlib as mpl
    mpl.rcParams['xtick.labelsize'] = 24
    mpl.rcParams['ytick.labelsize'] = 24
    '''Overplot pre- and post-adjusted disc UVTable data'''
    print('    plotting disc adjustments')
    fig = plt.figure()
    plt.suptitle(disc+'\n'+r'fitted inc. %.3f$^o$, PA %.3F$^o$, dRA %.6f", dDec %.6f"'%(Inc * 180 / np.pi, Pa * 180 / np.pi, Dra * rad_to_arcsec, Ddec * rad_to_arcsec))
    gs = GridSpec(2, 2, hspace=0, left=.1)
    ax = fig.add_subplot(gs[0, 0])
    #plt.plot(uv_pre.u, uv_pre.v, '.', label='pre-correction')
    plt.plot(uv_post.u, uv_post.v, '+', label='post')
    plt.xlabel('u [m]', fontsize=24)
    plt.ylabel('v [m]', fontsize=24)
    #plt.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    #plt.loglog(np.hypot(uv_pre.u, uv_pre.v) / wle, uv_pre.re, '.')
    plt.loglog(np.hypot(uv_post.u, uv_post.v) / wle, uv_post.re, '+')
    plt.ylabel('Re(V) [Jy]', fontsize=24)
    plt.xlabel('Baseline [$\lambda$]', fontsize=24)
    '''
    ax3 = fig.add_subplot(gs[1, 1])
    plt.semilogy(np.hypot(uv_pre.u, uv_pre.v) / wle / 1e3, uv_pre.im, '.')
    plt.semilogy(np.hypot(uv_post.u, uv_post.v) / wle / 1e3, uv_post.im, '+')
    plt.xlabel('Baseline [k$\lambda$]', fontsize=24)
    plt.ylabel('Im(V) [Jy]', fontsize=24)
    '''
    plt.savefig(savedir + 'disc_transformations.png')
    plt.close()


def plot_snr(uv, binwidth, disc, savedir):
    print('  finding, plotting SNR of binned data')
    baselines = np.hypot(uv.u, uv.v)
    sortidx = np.argsort(baselines)
    bls = baselines[sortidx] / 1e3
    snr = abs(uv.re[sortidx]) / (1 / uv.weights[sortidx] ** .5)
    binwidth /= 1e3

    bins = np.arange(0, max(bls) + binwidth, binwidth)
    mids = bins[:-1] + binwidth / 2
    bins_snr = np.vstack((np.digitize(bls, bins), snr)).T
    mean_snr = [np.mean(bins_snr[bins_snr[:, 0] == i, 1]) for i in range(len(bins))] # Runtimewarning here; how to ignore?
    std_snr = [np.std(bins_snr[bins_snr[:, 0] == i, 1]) for i in range(len(bins))]

    plt.figure()
    plt.suptitle(disc)
    plt.subplot(211)
    plt.errorbar(mids, mean_snr[1:], yerr=std_snr[1:], fmt='r.', ecolor='#A4A4A4', label='Observations, %s k$\lambda$ bins'%binwidth)
    plt.axhline(1, c='k', ls='--', label='SNR = 1')
    plt.ylabel(r'SNR = |Re(V)| $\cdot$ $\sqrt{\rm weights}$', fontsize=24)
    plt.legend()

    plt.subplot(212)
    plt.errorbar(mids, mean_snr[1:], yerr=std_snr[1:], fmt='r.', ecolor='#A4A4A4', label='Observations, %s k$\lambda$ bins'%binwidth)
    plt.axhline(1, c='k', ls='--', label='SNR = 1')
    plt.yscale('log')
    plt.xlabel('Baseline [k$\lambda$]', fontsize=24)
    plt.ylabel(r'SNR = |Re(V)| $\cdot$ $\sqrt{\rm weights}$', fontsize=24)

    plt.savefig(savedir + 'binned_data_snr_%sklambda_bins.png'%binwidth)
    plt.close()


def plot_fit(disc, Rmax, baselines, uv, model, GPHF, mu, nbins, alpha, smooth_strength, uvbin_size, uv_pts_toplot, pi_toplot, mu_toplot, alphas_toplot, pointwise_sc_toplot, stop_criterion_toplot, enforce_positivity, smooth_iterations, cut, counter_gp, known_profile, image_extracted_profile, add_noise, nfits, cmaps, snr_plot_binwidth, plot_savefile, savedir):
    plt.figure()
    plt.subplot(222)
    for l in range(len(uv_pts_toplot)):
        plt.loglog(uv_pts_toplot[l], pi_toplot[l] * 2 * np.pi, c=cmaps[counter_gp](l / len(uv_pts_toplot)))
    plt.subplot(221)
    for l in range(len(mu_toplot)):
        plt.plot(GPHF.Rc * rad_to_arcsec, mu_toplot[l], c=cmaps[counter_gp](l / len(mu_toplot)))
    plt.subplot(245)
    plt.plot(alphas_toplot)
    plt.subplot(246)
    plt.semilogy(stop_criterion_toplot)
    plt.subplot(224)
    for l in range(len(uv_pts_toplot)):
        plt.loglog(uv_pts_toplot[l], pointwise_sc_toplot[l] * 2 * np.pi, c=cmaps[counter_gp](l / len(uv_pts_toplot)))
    plt.savefig(savedir+'_fit_iteration_diag.png')
    print('saved diag plot')

    grid = np.linspace(0, Rmax, 10000)

    plt.figure()
    plt.plot(GPHF.Rc[:cut] * rad_to_arcsec, mu[:cut], label=r'Fit, %s bins, $\alpha$ =%s, ss=%.0e'%(nbins,alpha,smooth_strength))
    if known_profile: plt.plot(grid * rad_to_arcsec, model(grid), 'k--', label='Input profile')
    plt.legend(fontsize=24, ncol=1)
    plt.savefig(savedir + 'fit_quick.png')

    gs1 = GridSpec(4, 2, bottom=.08, top=.96, left=.1, right=.97, hspace=0)
    gs1_small = GridSpec(8, 2, bottom=.08, top=.72, left=.1, right=.97, hspace=0)
    gs1_inset = GridSpec(8, 2, bottom=.08, top=.86, left=.1, right=.92, hspace=0)
    gs2 = GridSpec(4, 2)
    fig1 = plt.figure()

    ki = 2*np.pi*np.logspace(np.log10(min(baselines.min(), GPHF.uv_pts[0])),
                             np.log10(max(baselines.max(), GPHF.uv_pts[-1])),
                             10**4)

    ax1 = fig1.add_subplot(gs1[0])
    plt.title('enforce positive = %s'%enforce_positivity)
    plt.ylabel('I [Jy sr$^{-1}$]', fontsize=24)

    if known_profile:
        resid = model(GPHF.Rc[:cut]) - mu[:cut]
        rmse = (np.mean(resid ** 2)) ** .5
        if counter_gp == 0: plt.plot(grid * rad_to_arcsec, model(grid), 'k--', label='Input profile')

    if image_extracted_profile and counter_gp == 0:
        clean_r, clean_i, _  = np.genfromtxt(pwd + '/../ALMAsim/' + disc + '/' + disc + '_cleaned_image_radial_profile.txt').T
        interp_func = interp1d(clean_r, clean_i, kind='nearest', bounds_error=False)
        regrid_clean_i = interp_func(GPHF.Rc * rad_to_arcsec)[:-1]
        plt.plot(GPHF.Rc[:cut] * rad_to_arcsec, regrid_clean_i, c='#a4a4a4', label=r'CLEAN, Briggs 0.5')
        if known_profile:
            resid_clean = model(GPHF.Rc[:cut]) - regrid_clean_i
            rmse_clean = (np.mean(resid_clean ** 2)) ** .5

    #plt.errorbar(GPHF.Rc * rad_to_arcsec, mu, yerr=np.diag(cov_renorm)**0.5, fmt='+', label='bins', zorder=-1)
    #plt.fill_between(GPHF.Rc * rad_to_arcsec, mu - np.diag(cov_renorm)**0.5, mu + np.diag(cov_renorm)**0.5, where=  mu - np.diag(cov_renorm)**0.5 >=  mu + np.diag(cov_renorm)**0.5, facecolor='r', alpha=.1)
    if add_noise: plt.plot(GPHF.Rc[:cut] * rad_to_arcsec, mu[:cut], label=r'Fit, %s bins, $\alpha$ =%s, ss=%.0e,'%(nbins,alpha,smooth_strength)+'\nmean[added noise / Re(V)] =%.2e'%mean_ratio)
    else: plt.plot(GPHF.Rc[:cut] * rad_to_arcsec, mu[:cut], label=r'Fit, %s bins, $\alpha$ =%s, ss=%.0e'%(nbins,alpha,smooth_strength))
    plt.legend(fontsize=10, ncol=1)
    xlo1, xhi1 = ax1.get_xlim()
    plt.setp(ax1.get_xticklabels(), visible=False)


    ax7 = fig1.add_subplot(gs1[2])
    plt.ylabel('I [Jy sr$^{-1}$]', fontsize=24)

    if counter_gp == 0:
        if image_extracted_profile: plt.semilogy(GPHF.Rc[:cut] * rad_to_arcsec, regrid_clean_i, c='#a4a4a4', label=r'CLEAN, Briggs 0.5')
        if known_profile: plt.semilogy(grid * rad_to_arcsec, model(grid), 'k--', label='Input profile')
    ax7.semilogy(GPHF.Rc[:cut] * rad_to_arcsec, mu[:cut], label=r'Fit, %s bins, $\alpha$ =%s, ss=%.0e'%(nbins,alpha,smooth_strength))
    ylo, yhi = ax7.get_ylim()
    if ylo < 1e6: plt.ylim(bottom=1e6)
    plt.xlim(xlo1, xhi1)

    if known_profile:
        ax3 = fig1.add_subplot(gs1[4])
        plt.xlabel('r ["]', fontsize=24)
        plt.ylabel('Normalized residual', fontsize=24)

        plt.axhline(0, c='k', ls='--')

        plt.plot(GPHF.Rc[:cut] * rad_to_arcsec, resid[:cut] / max(model(GPHF.Rc[:cut])), '.', label=r'RMSE %.2e' % (rmse / max(model(GPHF.Rc[:cut]))))

        if counter_gp == 0 and image_extracted_profile:
            plt.plot(GPHF.Rc[:cut] * rad_to_arcsec, resid_clean / max(model(GPHF.Rc[:cut])), '.', c='#a4a4a4',
                     label=r'CLEAN, Briggs 0.5, RMSE %.2e'% (rmse_clean / max(model(GPHF.Rc[:cut]))))

        plt.legend(fontsize=24, ncol=2)
        plt.xlim(xlo1, xhi1)
        plt.setp(ax7.get_xticklabels(), visible=False)
    else: ax7.set_xlabel('r ["]', fontsize=24)

    ax5 = fig1.add_subplot(gs1[1])
    plt.ylabel('Re(V) [Jy]', fontsize=24)

    ax5_2 = fig1.add_subplot(gs1_inset[1])

    ax2 = fig1.add_subplot(gs1[3])
    plt.ylabel('Re(V) [Jy]', fontsize=24)

    if counter_gp == 0:
        #'''
        uvbins, binned_re, binned_im, binned_re_err, binned_im_err = \
            uv.plot(linestyle='+', color='#33D9FF',
                label=r'Obs., %.0f k$\lambda$ bins' % (uvbin_size / 1e3),
                uvbin_size=uvbin_size, fontsize=24, return_binned_values=True)
        '''
        print('\n3.1\n')
        print('baselines',baselines)
        nbinz = np.int(baselines.max() / uvbin_size)
        print('nbins',nbinz)
        counts, edges = np.histogram(baselines, bins=nbinz)
        means, edges = np.histogram(baselines, weights=uv.re, bins=nbinz)
        centres = 0.5 * (edges[1:] + edges[:-1])
        means /= np.maximum(counts, 1)
        print('\n3.2\n')
        ax5.plot(centres, means, 'x', alpha=.75)
        print('\n4\n')
        '''
        #ax5.semilogx(baselines, uv.re, marker='.', ls='none', c='k', label='Obs., unbinned')
        ax5.plot(uvbins, binned_re, '+', color='#33D9FF', label=r'Obs., %.0f k$\lambda$ bins'%(uvbin_size/1e3))

        ax2.plot(baselines, uv.re, marker='+', ls='none', c='k', label='Obs. > 0')
        ax2.plot(baselines, -uv.re, marker='x', ls='none', c='#A4A4A4', label='Obs. < 0')

        # plt.errorbar(baselines, uv.re, yerr=uv.weights**-0.5,marker='+', ls='none', c='k', label='obs')
        # plt.errorbar(baselines, -uv.re, yerr=uv.weights**-0.5,marker='x', ls='none', c='c')

        ax2.plot(uvbins, binned_re, marker='+', ls='none', color='#33D9FF', label=r'Obs. > 0, %.0f k$\lambda$ bins'%(uvbin_size/1e3))
        ax2.plot(uvbins, -binned_re, marker='x', ls='none', color='#9CE73C', label=r'Obs. < 0, %.0f k$\lambda$ bins'%(uvbin_size/1e3))
        #ax2.plot(centres, means, '+', color='g')
        #ax2.plot(centres, -means, 'x', color='y')

        #'''

        ax5_2.semilogx(uvbins, binned_re, '+', color='#33D9FF')
        ax5_2.set_xlim(1e6,3e6)
        ax5_2.set_ylim(1e6,3e6)

        uvbins, binned_re, binned_im, binned_re_err, binned_im_err = \
            uv.plot(linestyle='x', linestyle2='+', color='r',
                label=r'obs, %.0f k$\lambda$ bins' % (uvbin_size * 50 / 1e3),
                uvbin_size=uvbin_size * 50, fontsize=24, return_binned_values=True)
        '''
        nbinz = np.int(baselines.max() / uvbin_size / 50)
        print('nbins',nbinz)
        counts, edges = np.histogram(baselines, bins=nbinz)
        means, edges = np.histogram(baselines, weights=uv.re, bins=nbinz)
        centres = 0.5 * (edges[1:] + edges[:-1])
        means /= np.maximum(counts, 1)
        print('\n3.2\n')
        print('\n4\n')
        '''
        ax2.plot(uvbins, binned_re, marker='+', ls='none', color='#0032FF', label=r'Obs. > 0, %.0f k$\lambda$ bins'%(uvbin_size * 50 / 1e3))
        ax2.plot(uvbins, -binned_re, marker='x', ls='none', color='#1A9B05', label=r'Obs. < 0, %.0f k$\lambda$ bins'%(uvbin_size * 50 / 1e3))
        #ax2.plot(centres, means, '+', color='k')
        #ax2.plot(centres, -means, 'x', color='b')

        #ax2.errorbar(uvbins, binned_re, yerr=binned_re_err, marker='+', color='r', label=r'Obs. > 0, %.0f k$\lambda$ bins'%(uvbin_size * 50 / 1e3))
        #ax2.errorbar(uvbins, -binned_re, yerr=binned_re_err, marker='x', color='m', label=r'Obs. < 0, %.0f k$\lambda$ bins'%(uvbin_size * 50 / 1e3))
        #'''
        ax2.legend(fontsize=10)

    #plt.errorbar(ki / (2 * np.pi), GPHF.HankelTransform(ki), yerr=np.diag(cov_renorm)**0.5, fmt='+', label='bins', zorder=-1)
    ax5.semilogx(ki / (2 * np.pi), GPHF.HankelTransform(ki))#, label=r'%s bins, $\alpha$ =%s, ss=%.0e' % (nbins, alpha, smooth_strength))

    ax5_2.semilogx(ki / (2 * np.pi), GPHF.HankelTransform(ki))

    ax5.legend(fontsize=10)
    xlo, xhi = ax5.get_xlim()
    plt.setp(ax5.get_xticklabels(), visible=False)

    ax2.set_ylim(bottom=1e-7)
    ax2.set_ylim(top=1)
    ax2.set_xlim(xlo, xhi)
    plt.setp(ax2.get_xticklabels(), visible=False)

    plot_log_abs(ki/(2*np.pi), GPHF.HankelTransform(ki))#, alpha=.5)#, label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins,alpha,smooth_strength))


    ax4 = fig1.add_subplot(gs1[5])
    plt.ylabel('Power', fontsize=24)

    #gs_cbar = GridSpec(2, 8, right=.99)
    plt.loglog(uv_pts_toplot[0], pi_toplot[0] * 2 * np.pi, c=cmaps[counter_gp](0), label=r'%s bins, $\alpha$ =%s, ss=%.0e, %s iterations'%(nbins, alpha, smooth_strength, smooth_iterations))
    for l in range(len(uv_pts_toplot)):
        plt.loglog(uv_pts_toplot[l], pi_toplot[l] * 2 * np.pi, c=cmaps[counter_gp](l / len(uv_pts_toplot)))
    plt.xlim(xlo, xhi)
    plt.legend(fontsize=10)
    plt.setp(ax4.get_xticklabels(), visible=False)

    '''
    ax6 = fig1.add_subplot(gs1_small[12])
    plt.ylabel(r'$\chi^2$', fontsize=24)

    #plt.plot(np.array(chi2s_alt) / len(uv.u), '.', label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
    plt.plot(np.array(chi2s) / len(uv.u), label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
    #plt.legend(fontsize=10)

    plt.setp(ax6.get_xticklabels(), visible=False)


    ax9 = fig1.add_subplot(gs1_small[14])
    plt.xlabel('Fit iteration #', fontsize=24)
    plt.ylabel('Con.#(i$_{cov}$)', fontsize=24)

    plt.semilogy(condition_numbers, label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
    plt.yticks(ax9.get_ylim(), ('%.1e'%ax9.get_ylim()[0], '%.1e'%ax9.get_ylim()[1]))
    '''

    ax8 = fig1.add_subplot(gs1[7])
    plt.ylabel(r'SNR = |Re(V)| $\cdot$ $\sqrt{\rm w}$', fontsize=24)
    plt.xlabel(r'Baseline [$\lambda$]', fontsize=24)
    plt.xlim(xlo, xhi)

    if counter_gp == 0:
        sortidx = np.argsort(baselines)
        bls = baselines[sortidx] / 1e3
        snr = abs(uv.re[sortidx]) / (1 / uv.weights[sortidx] ** .5)
        binwidth = snr_plot_binwidth / 1e3

        bins = np.arange(0, max(bls) + binwidth, binwidth)
        mids = bins[:-1] + binwidth / 2
        bins_snr = np.vstack((np.digitize(bls, bins), snr)).T
        mean_snr = [np.mean(bins_snr[bins_snr[:, 0] == i, 1]) for i in
                    range(len(bins))]  # Runtimewarning here; how to ignore?
        std_snr = [np.std(bins_snr[bins_snr[:, 0] == i, 1]) for i in range(len(bins))]

        plt.errorbar(mids * 1e3, mean_snr[1:], yerr=std_snr[1:], color='#33D9FF',fmt='.', ecolor='#A4A4A4',
                     label='%.0f k$\lambda$' % binwidth)


        binwidth = snr_plot_binwidth * 50 / 1e3
        bins = np.arange(0, max(bls) + binwidth, binwidth)
        mids = bins[:-1] + binwidth / 2
        bins_snr = np.vstack((np.digitize(bls, bins), snr)).T
        mean_snr = [np.mean(bins_snr[bins_snr[:, 0] == i, 1]) for i in
                    range(len(bins))]  # Runtimewarning here; how to ignore?
        std_snr = [np.std(bins_snr[bins_snr[:, 0] == i, 1]) for i in range(len(bins))]

        plt.errorbar(mids * 1e3, mean_snr[1:], yerr=std_snr[1:], color='r',fmt='.', ecolor='#9CE73C', alpha=.5,
                     label='Obs., %.0f k$\lambda$ bins' % binwidth, zorder=10)


        plt.axhline(1, c='k', ls='--', label='SNR = 1')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(fontsize=24)

    '''
    plt.axhline(0, c='#A4A4A4', ls='--')
    plt.xlabel('Baseline [$\lambda$]', fontsize=24)
    plt.ylabel('Im(V) [Jy]', fontsize=24)
    if counter_gp == 0:
        plt.errorbar(uvbins, binned_im, yerr=binned_im_err, marker='.', ls='none', c='k', label='obs, %.0f k$\lambda$ bins'%(uvbin_size/1e3))
    ax8.set_xscale('log')
    plt.xlim(xlo, xhi)
    plt.legend(fontsize=10)
    '''

    '''
    corr = cov_renorm.copy()
    for i in range(len(mu)):
        corr[i] /= np.sqrt(cov_renorm[i, i])
        corr[:, i] /= np.sqrt(cov_renorm[i, i])
    '''

    if counter_gp == nfits - 1:
        plt.savefig(plot_savefile)
        plt.close()

def plot_msfigs(plot_msfigs_savefile, disc, Rmax, baselines, uv, model, GPHF, mu, nbins, alpha, smooth_strength, uvbin_size, uv_pts_toplot, pi_toplot, enforce_positivity, smooth_iterations, cut, counter_gp, known_profile, image_extracted_profile, add_noise, nfits, cmaps, snr_plot_binwidth, plot_savefile, savedir):
        #plt.ion()
        import matplotlib as mpl
        mpl.rcParams['xtick.labelsize'] = 24
        mpl.rcParams['ytick.labelsize'] = 24

        grid = np.linspace(0, Rmax, 10000)
        plt.figure()
        plt.plot(GPHF.Rc[:cut] * rad_to_arcsec, mu[:cut], label=r'Fit, %s bins, $\alpha$ =%s, ss=%.0e'%(nbins,alpha,smooth_strength))
        if known_profile: plt.plot(grid * rad_to_arcsec, model(grid), '#FF0000', label='Input profile')
        plt.legend(fontsize=24, ncol=1)
        plt.savefig(savedir + 'fit_quick.png')

        print('making ms figs')
        grid = np.linspace(0, Rmax, 10000)

        #gs1_single = GridSpec(4, 2, bottom=.18, top=.98, left=.05, right=.97, hspace=0)
        gs1_single = GridSpec(2, 2, bottom=.1, top=.98, left=.08, right=.98, hspace=0)
        gs1_single2 = GridSpec(2, 2, bottom=.085, top=.7, left=.08, right=.98, hspace=0)
        gs1_single_offset = GridSpec(3, 2, bottom=-0.07, top=.83, left=.07, right=.98, hspace=0)
        gs1_big = GridSpec(1, 2, bottom=.25, top=.98, left=.08, right=.98, hspace=0)
        gs1_big2 = GridSpec(1, 2, bottom=.085, top=.75, left=.08, right=.98, hspace=0)
        gs1_double = GridSpec(6, 2, bottom=-.1, top=.98, left=.08, right=.98, hspace=0)
        gs1_double2 = GridSpec(5, 2, bottom=-.17, top=.98, left=.08, right=.98, hspace=0)
        gs1 = GridSpec(4, 2, bottom=.15, top=.98, left=.05, right=.97, hspace=0)
        gs1_2 = GridSpec(4, 2, bottom=.08, top=.89, left=.05, right=.97, hspace=0)
        gs1_small = GridSpec(8, 2, bottom=.08, top=.72, left=.05, right=.97, hspace=0)
        gs2 = GridSpec(4, 2)
        fig1 = plt.figure()

        ki = 2*np.pi*np.logspace(np.log10(min(baselines.min(), GPHF.uv_pts[0])),
                                 np.log10(max(baselines.max(), GPHF.uv_pts[-1])),
                                 10**4)

        #ax1 = fig1.add_subplot(gs1_big[0])
        ax1 = fig1.add_subplot(gs1_single2[0])
        #ax1.text(.95, .5, 'a)', transform=ax1.transAxes, fontsize=24)
        #ax1.text(.9, .5, 'a)', transform=ax1.transAxes, fontsize=24)
        ax1.set_ylabel('I [$10^{10}$ Jy sr$^{-1}$]', fontsize=24)
        #ax1.set_xlabel('r ["]', fontsize=24)

        #'''
        if known_profile:
            resid = model(GPHF.Rc[:cut]) - mu[:cut]
            rmse = (np.mean(resid ** 2)) ** .5
            if counter_gp == 0: plt.plot(grid * rad_to_arcsec, model(grid) / 1e10, c='#FF0000', label='Andrews+ 18')#label='Andrews et al. 2018 (for reference)')
            #"""
            if disc == 'dsharp_5e10_C437_40minuteintegration_rev3':
                ax1.hlines(y=3, xmin=0.07, xmax=0.1617, color='k')
                ax1.text(.08, 2.75, r'$\theta$', fontsize=24)
            #"""
            if disc == 'as209':
                ax1.hlines(y=3.7, xmin=0.1, xmax=0.135, color='k')
                ax1.text(.09, 3., r'$\theta$', fontsize=24)
            if disc == 'synth_richard_profile_original':
                ax1.hlines(y=.08, xmin=0.1, xmax=0.233, color='k')
                ax1.text(.13, .03, r'$\theta_{\rm beam}$', fontsize=24)
        #'''

        if image_extracted_profile and disc == 'dr_tau':
            ax1.hlines(y=6, xmin=0.1, xmax=0.22, color='k')
            ax1.text(.155, 4.75, r'$\theta$', fontsize=24)

        if known_profile and disc == 'double_gauss_rev3':
            gauss_r, gauss_i = np.genfromtxt(savedir + 'gauss_fit.txt').T
            ring_r, ring_i = np.genfromtxt(savedir + 'gauss_ring_fit.txt').T
            gauss_i /= 1e10
            ring_i /= 1e10
            #from scipy.interpolate import interp1d
            #def gauss_fit(r):
            #    interp_f_gauss = interp1d(gauss_r, gauss_i, kind='linear', bounds_error=False, fill_value=(gauss_r[0], 0))
            #    return interp_f(r)
            plt.plot(gauss_r, gauss_i + ring_i, 'c-.', label='Sum of individual fits', zorder=20)

        if image_extracted_profile and counter_gp == 0:
            ##clean_r, clean_i, _  = np.genfromtxt(pwd + '/../ALMAsim/' + disc + '/' + disc + '_cleaned_image_radial_profile.txt').T
            clean_r, clean_i  = np.genfromtxt(pwd + '/dr_tau_longetal19.csv').T
            clean_r = np.linspace(.01,.31,1e3)
            clean_i = .2 * .02111 / (.13 * .1) * 4.25e10 / 1e10  * (clean_r/.267)**-.7 * np.exp(-(clean_r/.267)**5.37)
            interp_func = interp1d(clean_r, clean_i, kind='nearest', bounds_error=False)
            regrid_clean_i = interp_func(GPHF.Rc * rad_to_arcsec)[:-1]
            plt.plot(GPHF.Rc[:cut] * rad_to_arcsec, regrid_clean_i / 1e10, c='#FF0000', label=r'DSHARP, Briggs 0.5')
            #plt.plot(clean_r, clean_i, c='#FF0000', label=r'Long +19, tapered power law')
            from scipy.integrate import trapz
            print('integral',trapz(clean_i, clean_r))
            if known_profile:
                resid_clean = model(GPHF.Rc[:cut]) - regrid_clean_i
                rmse_clean = (np.nanmean(resid_clean ** 2)) ** .5 # NaNs arise
                # when this profile doesn't extend to as large radii as the fit

        #"""
        if add_noise: plt.plot(GPHF.Rc[:cut] * rad_to_arcsec, mu[:cut], label=r'Fit, %s bins, $\alpha$ =%s, ss=%.0e,'%(nbins,alpha,smooth_strength)+'\nmean[added noise / Re(V)] =%.2e'%mean_ratio)
        else: plt.plot(GPHF.Rc[:cut] * rad_to_arcsec, mu[:cut] / 1e10, c='#00FFFF', label='Frankenstein')#label=r'Fit to Fedele et al. 2018 data') #AD13FF
        #"""

        #plt.errorbar(GPHF.Rc * rad_to_arcsec, mu, yerr=np.diag(cov_renorm)**0.5, fmt='+', label='bins', zorder=-1)
        #plt.fill_between(GPHF.Rc * rad_to_arcsec, mu - np.diag(cov_renorm)**0.5, mu + np.diag(cov_renorm)**0.5, where=  mu - np.diag(cov_renorm)**0.5 >=  mu + np.diag(cov_renorm)**0.5, facecolor='r', alpha=.1)
        plt.legend(fontsize=24, ncol=1)
        #plt.legend(ncol=1, loc=[.4, .05])
        xlo1, xhi1 = ax1.get_xlim()

        #"""
        plt.setp(ax1.get_xticklabels(), visible=False)
        #"""
        #plt.xlim(xlo1, 1.35)
        #plt.xlim(xlo1, .31)
        #"""
        ax7 = fig1.add_subplot(gs1_single2[2])
        plt.plot(GPHF.Rc[:cut] * rad_to_arcsec, mu[:cut], c='#00FFFF', label='Frankenstein')
        ax7.set_yscale('log')
        #'''
        #ax7.text(.95, .85, 'b)', transform=ax7.transAxes, fontsize=24)
        #ax7.text(.9, .7, 'b)', transform=ax7.transAxes, fontsize=24)
        plt.ylabel('I [Jy sr$^{-1}$]', fontsize=24)

        if counter_gp == 0:
            #ax7.semilogy(GPHF.Rc[:cut] * rad_to_arcsec, mu[:cut], c='#AD13FF')
            if known_profile: plt.plot(grid * rad_to_arcsec, model(grid), c='#FF0000')
            if image_extracted_profile: plt.plot(GPHF.Rc[:cut] * rad_to_arcsec, regrid_clean_i, c='#FF0000')#c='#a4a4a4')
        ylo, yhi = ax7.get_ylim()
        if ylo < 1e6: plt.ylim(bottom=1.1e6)
        plt.xlim(xlo1, xhi1)
        #"""
        """
        #'''
        if known_profile:
            #ax3 = fig1.add_subplot(gs1_single[4]) #[2])
            ax3 = fig1.add_subplot(gs1_double[8]) #[2])
            #ax3.text(.1, .8, 'c)', transform=ax3.transAxes, fontsize=24)
            #ax3.text(.95, .1, 'c)', transform=ax3.transAxes, fontsize=24)
            plt.xlabel('r ["]', fontsize=24)
            plt.ylabel('Norm. resid.', fontsize=24)
            #plt.ylabel('Normalized difference', fontsize=24)

            plt.axhline(0, c='k', ls='--', zorder=10)

            plt.plot(GPHF.Rc[:cut] * rad_to_arcsec, resid[:cut] / max(model(GPHF.Rc[:cut])), '.', c='#AD13FF', label=r'Fit, RMSE %.3f' % round(rmse / max(model(GPHF.Rc[:cut])),3))
            #plt.plot(GPHF.Rc[:cut] * rad_to_arcsec, resid[:cut] / max(model(GPHF.Rc[:cut])), '.', label=r'RMS difference %.2f' % (rmse / max(model(GPHF.Rc[:cut]))))

            if counter_gp == 0 and image_extracted_profile:
                plt.plot(GPHF.Rc[:cut] * rad_to_arcsec, resid_clean / max(model(GPHF.Rc[:cut])), '.', c='#FF0000',
                         label='CLEAN, RMSE %.3f'% round(rmse_clean / max(model(GPHF.Rc[:cut])),3))

            #plt.legend(fontsize=24, ncol=1)
            plt.legend(fontsize=24, ncol=1, loc=('upper right'))
            plt.xlim(xlo1, xhi1)
            #plt.xlim(xlo1, 1.21)
            #plt.setp(ax7.get_xticklabels(), visible=False)
        #else: ax7.set_xlabel('r ["]', fontsize=24)
        #'''
        """
        ax7.set_xlabel('r ["]', fontsize=24)
        #"""
        #ax5 = fig1.add_subplot(gs1_2[4])
        #ax5 = fig1.add_subplot(gs1_double[1])
        ax5 = fig1.add_subplot(gs1_double2[1])
        #ax5 = fig1.add_subplot(gs1_big[1])
        #ax5.text(.95, .5, 'c)', transform=ax5.transAxes, fontsize=24)
        #plt.ylabel('Re(V) [Jy]', fontsize=24)

        #"""
        #ax2 = fig1.add_subplot(gs1_2[6])
        #ax2 = fig1.add_subplot(gs1_single2[1])
        ax2 = fig1.add_subplot(gs1_big2[1])
        #ax2.text(.95, .9, 'd)', transform=ax2.transAxes, fontsize=24)
        #ax2.text(.25, .1, 'e)', transform=ax2.transAxes, fontsize=24)
        #plt.yticks([1e-5,1e-3,1e-1])
        plt.ylabel('Re(V) [Jy]', fontsize=24)
        #plt.xlabel(r'Baseline [$\lambda$]', fontsize=24)
        print('doing uv binning')
        if counter_gp == 0:
            #'''
            uvbins, binned_re, binned_im, binned_re_err, binned_im_err = \
                uv.plot(linestyle='+', color='#A4A4A4',
                    label=r'Obs, %.0f k$\lambda$ bins' % (uvbin_size),
                    uvbin_size=uvbin_size * 10, fontsize=24, return_binned_values=True)
            '''
            print('\n3.1\n')
            print('baselines',baselines)
            nbinz = np.int(baselines.max() / uvbin_size)
            print('nbins',nbinz)
            counts, edges = np.histogram(baselines, bins=nbinz)
            means, edges = np.histogram(baselines, weights=uv.re, bins=nbinz)
            centres = 0.5 * (edges[1:] + edges[:-1])
            means /= np.maximum(counts, 1)
            print('\n3.2\n')
            ax5.plot(centres, means, 'x', alpha=.75)
            print('\n4\n')
            '''

            #ax5.semilogx(baselines, uv.re, marker='.', ls='none', c='k', label='Obs., unbinned')
            ax5.plot(uvbins, binned_re, '+', color='#FF0000', label=r'Observations, %.0f k$\lambda$ bins'%(uvbin_size/1e3))

            ax2.plot(baselines, uv.re, marker='x', ls='none', c='#A4A4A4', label='Obs. > 0') #k
            ax2.plot(baselines, -uv.re, marker='x', ls='none', c='#A4A4A4', label='< 0')

            # plt.errorbar(baselines, uv.re, yerr=uv.weights**-0.5,marker='+', ls='none', c='k', label='obs')
            # plt.errorbar(baselines, -uv.re, yerr=uv.weights**-0.5,marker='x', ls='none', c='c')

            ax2.plot(uvbins, binned_re, marker='+', ls='none', color='#FF0000', label=r'> 0, %.0f k$\lambda$ bins'%(uvbin_size/1e3)) #33D9FF
            ax2.plot(uvbins, -binned_re, marker='+', ls='none', color='#FF0000', label=r'< 0, %.0f k$\lambda$'%(uvbin_size/1e3)) #9CE73C
            #ax2.plot(centres, means, '+', color='g')
            #ax2.plot(centres, -means, 'x', color='y')

            #'''
            uvbins2, binned_re2, binned_im2, binned_re_err2, binned_im_err2 = \
                uv.plot(linestyle='x', linestyle2='+', color='#A4A4A4',
                    label=r'obs, %.0f k$\lambda$ bins' % (uvbin_size * 50),
                    uvbin_size=uvbin_size * 50, fontsize=24, return_binned_values=True)
            '''
            nbinz = np.int(baselines.max() / uvbin_size / 50)
            print('nbins',nbinz)
            counts, edges = np.histogram(baselines, bins=nbinz)
            means, edges = np.histogram(baselines, weights=uv.re, bins=nbinz)
            centres = 0.5 * (edges[1:] + edges[:-1])
            means /= np.maximum(counts, 1)
            print('\n3.2\n')
            print('\n4\n')
            '''
            ##ax5.plot(uvbins2, binned_re2, '+', color='#FF8213', ms=6, label=r'%.0f k$\lambda$'%(uvbin_size * 50 / 1e3))

            ##ax2.plot(uvbins2, binned_re2, marker='+', ls='none', color='#FF8213', label=r'> 0, %.0f k$\lambda$'%(uvbin_size * 50 / 1e3))
            ##ax2.plot(uvbins2, -binned_re2, marker='x', ls='none', color='#FF8213', label=r'< 0, %.0f k$\lambda$'%(uvbin_size * 50 / 1e3)) #880901
            #ax2.plot(centres, means, '+', color='k')
            #ax2.plot(centres, -means, 'x', color='b')

            #ax2.errorbar(uvbins, binned_re, yerr=binned_re_err, marker='+', color='r', label=r'Obs. > 0, %.0f k$\lambda$ bins'%(uvbin_size * 50 / 1e3))
            #ax2.errorbar(uvbins, -binned_re, yerr=binned_re_err, marker='x', color='m', label=r'Obs. < 0, %.0f k$\lambda$ bins'%(uvbin_size * 50 / 1e3))
            #'''
            ##ax2.legend(fontsize=24, ncol=1)
            #ax2.legend(fontsize=24, ncol=1)

        #plt.errorbar(ki / (2 * np.pi), GPHF.HankelTransform(ki), yerr=np.diag(cov_renorm)**0.5, fmt='+', label='bins', zorder=-1)
        ##ax5.plot(ki / (2 * np.pi), GPHF.HankelTransform(ki), c='#00FFFF', lw=3)#c='#AD13FF')#, label=r'%s bins, $\alpha$ =%s, ss=%.0e' % (nbins, alpha, smooth_strength))
        #ax5.legend(fontsize=24)
        xlo, xhi = [9e3, 1e7] #ax5.get_xlim()
        plt.setp(ax5.get_xticklabels(), visible=False)

        '''
        galario_r, galario_i  = np.genfromtxt(pwd + '/dr_tau_galario.csv').T
        galario_r *= 1e3
        galario_i /= 1e3
        print('r,i',galario_r,galario_i)
        ax5.semilogx(galario_r[:-1], galario_i[:-1], c='#FF0000')
        '''

        ax2.set_ylim(bottom=1.1e-7)
        ax2.set_ylim(top=.9)
        ax5.set_xlim(xlo, xhi)
        ax5.set_ylim(-.025, .025)
        ax2.set_xlim(xlo, xhi)

        #ax2.set_xlim(xlo, 4e6)
        #ax5.set_xlim(xlo, 4e6)
        #ax2.set_xlim(xlo, 1e7)
        #ax5.set_xlim(-50, 2420)

        ##plt.setp(ax2.get_xticklabels(), visible=False)
        #ax5.set_xlabel(r'Baseline [$\lambda$]', fontsize=24)
        ax5.set_ylabel('Re(V) [Jy]', fontsize=24)
        ax2.set_xlabel(r'Baseline [$\lambda$]', fontsize=24)
        ax5.set_xscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')

        ##plot_log_abs(ki/(2*np.pi), GPHF.HankelTransform(ki), c='#00FFFF')#c='#AD13FF')#, alpha=.5)#, label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins,alpha,smooth_strength))
        #"""
        """
        ax10 = fig1.add_subplot(gs1_double[7])
        ax10.text(.95, .1, 'e)', transform=ax10.transAxes, fontsize=24)
        #ax8.text(.1, .1, 'f)', transform=ax8.transAxes, fontsize=24)
        plt.ylabel(r'Residual [Jy]', fontsize=24)
        plt.setp(ax10.get_xticklabels(), visible=False)
        ax10.set_xlim(xlo, xhi)

        ax10.set_xlim(xlo, 4e6)
        #ax10.set_xlim(xlo, 1e7)

        #ax9.set_ylim(1e-3, 1e1)
        snr_barrier = 5e5
        def find_nearest(array,value):
        	idx = (np.abs(array - value)).argmin()
        	#print 'nearest value %s. array position %s'%(array[idx],idx)
        	return idx

        resid_vis = binned_re - GPHF.HankelTransform(uvbins * 2 * np.pi)
        norm_resid_vis = (binned_re - GPHF.HankelTransform(uvbins * 2 * np.pi)) / binned_re
        print('find_nearest',find_nearest(uvbins, snr_barrier))
        resid_vis_to_barrier = resid_vis[:find_nearest(uvbins, snr_barrier)]
        rmse_vis = (np.mean(resid_vis_to_barrier ** 2)) ** .5
        norm_resid_vis_to_barrier = norm_resid_vis[:find_nearest(uvbins, snr_barrier)]
        norm_rmse_vis = (np.mean(norm_resid_vis_to_barrier ** 2)) ** .5
        print('norm_resid_vis',norm_resid_vis)
        #resid_vis *= 1e4
        #ax10.semilogx(uvbins, resid_vis, '.', c='#33D9FF')
        ax10.plot(uvbins, resid_vis, '+', c='#33D9FF')#, label='Obs. > 0, %.0f k$\lambda$'%(uvbin_size/1e3))
        ax10.plot(uvbins, -resid_vis, 'x', c='#9CE73C')#, label='< 0, %.0f k$\lambda$'%(uvbin_size/1e3))
        ax10.set_xscale('log')
        ax10.set_yscale('log')
        """
        '''
        ax4 = fig1.add_subplot(gs1_single[5])
        resid = binned_re - GPHF.HankelTransform(uvbins)
        rmse = (np.mean(resid ** 2)) ** .5
        ax4.semilogx(uvbins, (binned_re - GPHF.HankelTransform(uvbins)) / max(binned_re), label='RMSE %.2d'% (rmse / max(binned_re)))
        ax4.set_ylabel('Normalized residual', fontsize=24)
        xlo, xhi = ax4.get_xlim()
        ax4.set_xlim(xlo, xhi)
        #plt.legend(fontsize=10)
        plt.setp(ax4.get_xticklabels(), visible=False)
        '''
        '''
        ax4 = fig1.add_subplot(gs1_double[9])
        plt.ylabel('Power', fontsize=24)

        #gs_cbar = GridSpec(2, 8, right=.99)
        plt.loglog(uv_pts_toplot[0], pi_toplot[0] * 2 * np.pi, c=cmaps[counter_gp](0))#, label=r'%s bins, $\alpha$ =%s, ss=%.0e, %s iterations'%(nbins, alpha, smooth_strength, smooth_iterations))
        for l in range(len(uv_pts_toplot)):
            plt.loglog(uv_pts_toplot[l], pi_toplot[l] * 2 * np.pi, c=cmaps[counter_gp](l / len(uv_pts_toplot)))
        plt.xlim(xlo, xhi)

        plt.xlim(xlo, 1e7)

        #plt.legend(fontsize=10)
        plt.setp(ax4.get_xticklabels(), visible=False)
        '''
        '''
        ax6 = fig1.add_subplot(gs1_small[12])
        plt.ylabel(r'$\chi^2$', fontsize=24)

        #plt.plot(np.array(chi2s_alt) / len(uv.u), '.', label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        plt.plot(np.array(chi2s) / len(uv.u), label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        #plt.legend(fontsize=10)

        plt.setp(ax6.get_xticklabels(), visible=False)


        ax9 = fig1.add_subplot(gs1_small[14])
        plt.xlabel('Fit iteration #', fontsize=24)
        plt.ylabel('Con.#(i$_{cov}$)', fontsize=24)

        plt.semilogy(condition_numbers, label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        plt.yticks(ax9.get_ylim(), ('%.1e'%ax9.get_ylim()[0], '%.1e'%ax9.get_ylim()[1]))
        '''
        #'''
        """
        #ax8 = fig1.add_subplot(gs1_single[7])
        #ax8 = fig1.add_subplot(gs1_single[5])
        #ax8 = fig1.add_subplot(gs1_double[9])
        ax8 = fig1.add_subplot(gs1_single2[3])
        #ax8.text(.95, .8, 'f)', transform=ax8.transAxes, fontsize=24)
        #ax8.text(.1, .1, 'f)', transform=ax8.transAxes, fontsize=24)
        #plt.ylabel(r'SNR = |Re(V)| $\cdot$ $\sqrt{\rm w}$', fontsize=24)
        plt.ylabel(r'SNR', fontsize=24)
        plt.xlabel(r'Baseline [$\lambda$]', fontsize=24)
        xlo, xhi = ax1.get_xlim()
        ax8.set_xlim(xlo, xhi)

        #ax8.set_xlim(xlo, 4e6)
        #ax8.set_xlim(xlo, 1e7)

        #ax8.set_ylim(1e-1, 1e1)
        #ax8.set_ylim(1e-3, 1e1)
        ax8.set_ylim(0, 5)
        print('start snr')
        if counter_gp == 0:
            #'''
            sortidx = np.argsort(baselines)
            bls = baselines[sortidx] / 1e3
            snr = abs(uv.re[sortidx]) / (1 / uv.weights[sortidx] ** .5)
            binwidth = snr_plot_binwidth / 1e3

            bins = np.arange(0, max(bls) + binwidth, binwidth)
            mids = bins[:-1] + binwidth / 2
            bins_snr = np.vstack((np.digitize(bls, bins), snr)).T
            mean_snr = [np.mean(bins_snr[bins_snr[:, 0] == i, 1]) for i in
                        range(len(bins))]  # Runtimewarning here; how to ignore?
            std_snr = [np.std(bins_snr[bins_snr[:, 0] == i, 1]) for i in range(len(bins))]

            plt.errorbar(mids * 1e3, mean_snr[1:], yerr=std_snr[1:], color='#00FF6C',fmt='.', ecolor='#BDBDBD',
                         label='Obs., %.0f k$\lambda$ bins' % binwidth)


            binwidth = snr_plot_binwidth * 50 / 1e3
            bins = np.arange(0, max(bls) + binwidth, binwidth)
            mids = bins[:-1] + binwidth / 2
            bins_snr = np.vstack((np.digitize(bls, bins), snr)).T
            mean_snr = [np.mean(bins_snr[bins_snr[:, 0] == i, 1]) for i in
                        range(len(bins))]  # Runtimewarning here; how to ignore?
            std_snr = [np.std(bins_snr[bins_snr[:, 0] == i, 1]) for i in range(len(bins))]

            plt.errorbar(mids * 1e3, mean_snr[1:], yerr=std_snr[1:], color='r',fmt='.', ecolor='#00AAFF', alpha=.5,
                         label='%.0f k$\lambda$' % binwidth, zorder=10)
            #'''
            print('NB the bin centering is wrong; should use the results from uvplot')
            '''
            plt.errorbar(uvbins, binned_re, yerr=binned_re_err, color='#00FF6C',fmt='.', ecolor='#BDBDBD', alpha=.5,
                         label='%.0f k$\lambda$' % uvbin_size, zorder=10)
            plt.errorbar(uvbins2, binned_re2, yerr=binned_re_err2, color='r',fmt='.', ecolor='#00AAFF', alpha=.5,
                         label='%.0f k$\lambda$' % uvbin_size * 50, zorder=10)
            '''
            plt.axhline(1, c='k', ls='--', label='SNR = 1', zorder=10)
            plt.xscale('log')
            plt.legend(fontsize=24, loc='upper left')
            #plt.legend(fontsize=24, loc=[.5, .05])
        #'''
        """
        '''
        plt.axhline(0, c='#A4A4A4', ls='--')
        plt.xlabel('Baseline [$\lambda$]', fontsize=24)
        plt.ylabel('Im(V) [Jy]', fontsize=24)
        if counter_gp == 0:
            plt.errorbar(uvbins, binned_im, yerr=binned_im_err, marker='.', ls='none', c='k', label='obs, %.0f k$\lambda$ bins'%(uvbin_size/1e3))
        ax8.set_xscale('log')
        plt.xlim(xlo, xhi)
        plt.legend(fontsize=10)
        '''

        '''
        corr = cov_renorm.copy()
        for i in range(len(mu)):
            corr[i] /= np.sqrt(cov_renorm[i, i])
            corr[:, i] /= np.sqrt(cov_renorm[i, i])
        '''
        #"""
        eff_res = 5e6 #1.93e6 #2.45e6 #2.45e6
        ##ax5.axvline(eff_res, zorder=20, c='b', ls='--')#, label='Effective resolution')
        ##ax2.axvline(eff_res, zorder=20, c='b', ls='--')
        #ax2.text(1e6, .1, 'Effective\nresolution', fontsize=24, color='b')
        #ax10.axvline(eff_res, zorder=20, c='r', ls='--', label='RMSE[:eff. res.] %.3f'%round(rmse_vis,3))
        #ax8.axvline(eff_res, zorder=20, c='b', ls='--')
        #ax5.legend(loc=[.4,.3])
        #ax10.legend(fontsize=24, ncol=1)
        #"""
        plt.savefig(plot_msfigs_savefile)#, dpi=600)
        #plt.show()

def ms_f1(plot_msfigs_savefile, disc, Rmax, baselines, uv, model, GPHF, mu, nbins, alpha, smooth_strength, uvbin_size, uvbin_size2, uv_pts_toplot, pi_toplot, enforce_positivity, smooth_iterations, cut, counter_gp, known_profile, image_extracted_profile, add_noise, nfits, cmaps, snr_plot_binwidth, snr_plot_binwidth2, plot_savefile, savedir):
        #plt.ion()
        import matplotlib as mpl
        mpl.rcParams['xtick.labelsize'] = 24
        mpl.rcParams['ytick.labelsize'] = 24

        #gs1_single = GridSpec(4, 2, bottom=.18, top=.98, left=.05, right=.97, hspace=0)
        gs1_single = GridSpec(2, 2, bottom=.1, top=.98, left=.09, right=.98, hspace=0)
        gs1_single2 = GridSpec(2, 2, bottom=.085, top=.7, left=.09, right=.98, hspace=0)
        gs1_single_offset = GridSpec(3, 2, bottom=-0.07, top=.83, left=.07, right=.98, hspace=0)
        gs1_big = GridSpec(1, 2, bottom=.25, top=.98, left=.09, right=.98, hspace=0)
        gs1_big2 = GridSpec(1, 2, bottom=.085, top=.75, left=.09, right=.98, hspace=0)
        gs1_big3 = GridSpec(1, 2, bottom=.30875, top=.75625, left=.09, right=.98, hspace=0)
        gs1_double = GridSpec(4, 2, bottom=.085, top=.98, left=.09, right=.98, hspace=0)
        gs1_double_top = GridSpec(4, 2, bottom=.185, top=.98, left=.09, right=.98, hspace=0)
        gs1_double_mid = GridSpec(4, 2, bottom=-.115, top=1.08, left=.09, right=.98, hspace=0)
        gs1_double_bot = GridSpec(4, 2, bottom=.085, top=.88, left=.09, right=.98, hspace=0)
        gs1_double2 = GridSpec(5, 2, bottom=.085, top=.98, left=.09, right=.98, hspace=0)
        gs1 = GridSpec(4, 2, bottom=.15, top=.98, left=.05, right=.97, hspace=0)
        gs1_2 = GridSpec(4, 2, bottom=.08, top=.89, left=.05, right=.97, hspace=0)
        gs1_small = GridSpec(8, 2, bottom=.08, top=.72, left=.05, right=.97, hspace=0)
        gs2 = GridSpec(4, 2)

        fig1 = plt.figure()

        grid = np.linspace(0, Rmax, 10000)
        ki = 2*np.pi*np.logspace(np.log10(min(baselines.min(), GPHF.uv_pts[0])),
                                 np.log10(max(baselines.max(), GPHF.uv_pts[-1])),
                                 10**4)

        if disc == 'synth_gauss_rev3':
            ax1 = fig1.add_subplot(gs1_double[0])
        elif disc == 'asym_gauss_ring_rev3':
            ax1 = fig1.add_subplot(gs1_double[2])
        else:
            ax1 = fig1.add_subplot(gs1_double[4])

        resid = model(GPHF.Rc[:cut]) - mu[:cut]
        rmse = (np.mean(resid ** 2)) ** .5

        clean_r, clean_i, _  = np.genfromtxt(pwd + '/../ALMAsim/' + disc + '/' + disc + '_cleaned_image_radial_profile.txt').T
        interp_func = interp1d(clean_r, clean_i, kind='nearest', bounds_error=False)
        regrid_clean_i = interp_func(GPHF.Rc * rad_to_arcsec)[:-1]

        resid_clean = model(GPHF.Rc[:cut]) - regrid_clean_i
        rmse_clean = (np.nanmean(resid_clean ** 2)) ** .5 # NaNs arise
        # when this profile doesn't extend to as large radii as the fit

        if disc == 'synth_gauss_rev3':
            ax1.plot(grid * rad_to_arcsec, model(grid) / 1e10, 'k--', label='Input profile')#label='Andrews et al. 2018 (for reference)')
            ax1.plot(GPHF.Rc[:cut] * rad_to_arcsec, regrid_clean_i / 1e10, c='#a4a4a4', label=r'CLEAN, Briggs 0.5')
        else:
            ax1.plot(grid * rad_to_arcsec, model(grid) / 1e10, 'k--', label='Input')#label='Andrews et al. 2018 (for reference)')
            ax1.plot(GPHF.Rc[:cut] * rad_to_arcsec, regrid_clean_i / 1e10, c='#a4a4a4', label=r'CLEAN')
        ax1.plot(GPHF.Rc[:cut] * rad_to_arcsec, mu[:cut] / 1e10, c='r', label='Fit')#label=r'Fit to Fedele et al. 2018 data') #AD13FF

        if disc == 'synth_gauss_rev3' or disc == 'asym_gauss_ring_rev3':
            np.save(savedir + 'fit_temp.npy', [grid * rad_to_arcsec, model(grid) / 1e10, GPHF.Rc[:cut] * rad_to_arcsec, regrid_clean_i / 1e10, mu[:cut] / 1e10])

        #plt.errorbar(GPHF.Rc * rad_to_arcsec, mu, yerr=np.diag(cov_renorm)**0.5, fmt='+', label='bins', zorder=-1)
        #plt.fill_between(GPHF.Rc * rad_to_arcsec, mu - np.diag(cov_renorm)**0.5, mu + np.diag(cov_renorm)**0.5, where=  mu - np.diag(cov_renorm)**0.5 >=  mu + np.diag(cov_renorm)**0.5, facecolor='r', alpha=.1)

        if disc == 'double_gauss_rev3':
            g1_grid, g1_mod, g1_grid2, g1_clean, g1_mu = np.load(savedir + '../synth_gauss_rev3/fit_temp.npy').T
            g2_grid, g2_mod, g2_grid2, g2_clean, g2_mu = np.load(savedir + '../asym_gauss_ring_rev3/fit_temp.npy').T
            ax0_1 = fig1.add_subplot(gs1_double[0])
            ax0_2 = fig1.add_subplot(gs1_double[2])
            ax0_1.plot(g1_grid, g1_mod, 'k--', label='Input profile')#label='Andrews et al. 2018 (for reference)')
            ax0_1.plot(g1_grid2, g1_clean, c='#a4a4a4', label=r'CLEAN, Briggs 0.5')
            ax0_1.plot(g1_grid2, g1_mu, c='r', label='Fit')#label=r'Fit to Fedele et al. 2018 data') #AD13FF
            ax0_2.plot(g2_grid, g2_mod, 'k--', label='Input')#label='Andrews et al. 2018 (for reference)')
            ax0_2.plot(g2_grid2, g2_clean, c='#a4a4a4', label=r'CLEAN')
            ax0_2.plot(g2_grid2, g2_mu, c='r', label='Fit')#label=r'Fit to Fedele et al. 2018 data') #AD13FF
            ax0_1.text(.25, .8, 'a)', transform=ax0_1.transAxes, fontsize=24)
            ax0_2.text(.25, .8, 'b)', transform=ax0_2.transAxes, fontsize=24)
            ax0_1.legend(fontsize=20, ncol=1, loc='best')
            ax0_2.legend(fontsize=20, ncol=1, loc='best')
            ax0_1.set_ylabel('Brightness\n[$10^{10}$ Jy sr$^{-1}$]', fontsize=24)
            ax0_2.set_ylabel('Brightness\n[$10^{10}$ Jy sr$^{-1}$]', fontsize=24)
            plt.setp(ax0_1.get_xticklabels(), visible=False)
            plt.setp(ax0_2.get_xticklabels(), visible=False)

            gauss_r, gauss_i = np.genfromtxt(savedir + 'gauss_fit.txt').T
            ring_r, ring_i = np.genfromtxt(savedir + 'gauss_ring_fit.txt').T
            gauss_i /= 1e10
            ring_i /= 1e10
            #from scipy.interpolate import interp1d
            #def gauss_fit(r):
            #    interp_f_gauss = interp1d(gauss_r, gauss_i, kind='linear', bounds_error=False, fill_value=(gauss_r[0], 0))
            #    return interp_f(r)
            ax1.plot(gauss_r, gauss_i + ring_i, 'c-.', label='a)+b)', zorder=20)

        xlo1, xhi1 = ax1.get_xlim()

        ax1.legend(fontsize=20, ncol=1, loc='best')
        ax1.set_ylabel('Brightness\n[$10^{10}$ Jy sr$^{-1}$]', fontsize=24)
        plt.setp(ax1.get_xticklabels(), visible=False)

        if disc == 'synth_gauss_rev3':
            ax1.text(.35, .8, 'a)', transform=ax1.transAxes, fontsize=24)
        elif disc == 'asym_gauss_ring_rev3':
            ax1.text(.1, .8, 'b)', transform=ax1.transAxes, fontsize=24)
        else:
            ax1.text(.25, .8, 'c)', transform=ax1.transAxes, fontsize=24)

        if disc == 'double_gauss_rev3':
            ax3 = fig1.add_subplot(gs1_double[6])
            ax5 = fig1.add_subplot(gs1_double_top[1])
            ax2 = fig1.add_subplot(gs1_double_mid[3])
            ax10 = fig1.add_subplot(gs1_double_bot[5])
            ax8 = fig1.add_subplot(gs1_double_bot[7])

            uvbins, binned_re, binned_im, binned_re_err, binned_im_err = \
                uv.plot(linestyle='+', color='#A4A4A4',
                    label=r'Obs, %.0f k$\lambda$ bins' % (uvbin_size),
                    uvbin_size=uvbin_size, fontsize=24, return_binned_values=True)

            uvbins2, binned_re2, binned_im2, binned_re_err2, binned_im_err2 = \
                uv.plot(linestyle='x', linestyle2='+', color='#A4A4A4',
                    label=r'obs, %.0f k$\lambda$ bins' % (uvbin_size2),
                    uvbin_size=uvbin_size2, fontsize=24, return_binned_values=True)

            def find_nearest(array,value):
            	idx = (np.abs(array - value)).argmin()
            	#print 'nearest value %s. array position %s'%(array[idx],idx)
            	return idx

            snr_barrier = 5e5
            resid_vis = binned_re - GPHF.HankelTransform(uvbins * 2 * np.pi)
            norm_resid_vis = (binned_re - GPHF.HankelTransform(uvbins * 2 * np.pi)) / binned_re
            print('find_nearest',find_nearest(uvbins, snr_barrier))
            resid_vis_to_barrier = resid_vis[:find_nearest(uvbins, snr_barrier)]
            rmse_vis = (np.mean(resid_vis_to_barrier ** 2)) ** .5
            norm_resid_vis_to_barrier = norm_resid_vis[:find_nearest(uvbins, snr_barrier)]
            norm_rmse_vis = (np.mean(norm_resid_vis_to_barrier ** 2)) ** .5
            #print('norm_resid_vis',norm_resid_vis)

            sortidx = np.argsort(baselines)
            bls = baselines[sortidx] / 1e3
            snr = abs(uv.re[sortidx]) / (1 / uv.weights[sortidx] ** .5)
            binwidth = snr_plot_binwidth / 1e3

            bins = np.arange(0, max(bls) + binwidth, binwidth)
            mids = bins[:-1] + binwidth / 2
            bins_snr = np.vstack((np.digitize(bls, bins), snr)).T
            mean_snr = [np.mean(bins_snr[bins_snr[:, 0] == i, 1]) for i in
                        range(len(bins))]  # Runtimewarning here; how to ignore?
            std_snr = [np.std(bins_snr[bins_snr[:, 0] == i, 1]) for i in range(len(bins))]
            ax8.errorbar(mids * 1e3, mean_snr[1:], yerr=std_snr[1:], color='#a4a4a4',fmt='.', ecolor='k',
                         label='%.0f k$\lambda$ bins' % binwidth)

            binwidth = snr_plot_binwidth2 / 1e3
            bins = np.arange(0, max(bls) + binwidth, binwidth)
            mids = bins[:-1] + binwidth / 2
            bins_snr = np.vstack((np.digitize(bls, bins), snr)).T
            mean_snr = [np.mean(bins_snr[bins_snr[:, 0] == i, 1]) for i in
                        range(len(bins))]  # Runtimewarning here; how to ignore?
            std_snr = [np.std(bins_snr[bins_snr[:, 0] == i, 1]) for i in range(len(bins))]
            print('NB the bin centering for SNR plot is wrong; should use the results from uvplot')
            ax8.errorbar(mids * 1e3, mean_snr[1:], yerr=std_snr[1:], color='m',fmt='.', ecolor='#00AAFF', alpha=.5,
                         label='%.0f k$\lambda$' % binwidth, zorder=10)

            ax3.plot(GPHF.Rc[:cut] * rad_to_arcsec, resid_clean / max(model(GPHF.Rc[:cut])), '.', c='#a4a4a4',
                 label='CLEAN, RMSE %.3f'% round(rmse_clean / max(model(GPHF.Rc[:cut])),3))
            ax3.plot(GPHF.Rc[:cut] * rad_to_arcsec, resid[:cut] / max(model(GPHF.Rc[:cut])), '.', c='r', label=r'Fit, RMSE %.3f' % round(rmse / max(model(GPHF.Rc[:cut])),3))

            ax5.plot(uvbins, binned_re, '+', c='c', ms=8, label=r'Obs., %.0f k$\lambda$ bins'%(uvbin_size/1e3))
            ax2.plot(baselines, uv.re, marker='.', ls='none', c='k', label='Obs. > 0') #k
            ax2.plot(baselines, -uv.re, marker='.', ls='none', c='#A4A4A4', label='< 0')
            # plt.errorbar(baselines, uv.re, yerr=uv.weights**-0.5,marker='+', ls='none', c='k', label='obs')
            # plt.errorbar(baselines, -uv.re, yerr=uv.weights**-0.5,marker='x', ls='none', c='c')
            ax2.plot(uvbins, binned_re, marker='+', ms=8, ls='none', color='c', label=r'> 0, %.0f k$\lambda$ bins'%(uvbin_size/1e3)) #33D9FF
            ax2.plot(uvbins, -binned_re, marker='+', ms=8, ls='none', color='#65EB2F', label=r'< 0, %.0f k$\lambda$'%(uvbin_size/1e3)) #9CE73C

            #ax2.errorbar(uvbins, binned_re, yerr=binned_re_err, marker='+', color='r', label=r'Obs. > 0, %.0f k$\lambda$ bins'%(uvbin_size * 50 / 1e3))
            #ax2.errorbar(uvbins, -binned_re, yerr=binned_re_err, marker='x', color='m', label=r'Obs. < 0, %.0f k$\lambda$ bins'%(uvbin_size * 50 / 1e3))
            #plt.errorbar(ki / (2 * np.pi), GPHF.HankelTransform(ki), yerr=np.diag(cov_renorm)**0.5, fmt='+', label='bins', zorder=-1)
            ax5.plot(uvbins2, binned_re2, 'x', c='b', ms=8, label=r'%.0f k$\lambda$'%(uvbin_size2 / 1e3))
            ax5.plot(ki / (2 * np.pi), GPHF.HankelTransform(ki), c='r')#c='#AD13FF')#, label=r'%s bins, $\alpha$ =%s, ss=%.0e' % (nbins, alpha, smooth_strength))
            ax2.plot(uvbins2, binned_re2, marker='x', ms=8, ls='none', c='b', label=r'> 0, %.0f k$\lambda$'%(uvbin_size2 / 1e3))
            ax2.plot(uvbins2, -binned_re2, marker='x', ms=8, ls='none', c='m', label=r'< 0, %.0f k$\lambda$'%(uvbin_size2 / 1e3)) #880901
            plot_log_abs(ki/(2*np.pi), GPHF.HankelTransform(ki), ax2, c='r')#c='#AD13FF')#, alpha=.5)#, label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins,alpha,smooth_strength))

            ax10.plot(uvbins, norm_resid_vis, '+', ms=8, c='#F700FF', label='> 0, %.0f k$\lambda$'%(uvbin_size/1e3))
            ax10.plot(uvbins, -norm_resid_vis, 'x',  ms=8, c='#00FFAA', alpha=.5, label='< 0, %.0f k$\lambda$'%(uvbin_size/1e3))

            '''
            ax8.errorbar(uvbins, binned_re, yerr=binned_re_err, color='#00FF6C',fmt='.', ecolor='#BDBDBD', alpha=.5,
                         label='%.0f k$\lambda$' % uvbin_size, zorder=10)
            ax8.errorbar(uvbins2, binned_re2, yerr=binned_re_err2, color='r',fmt='.', ecolor='#00AAFF', alpha=.5,
                         label='%.0f k$\lambda$' % uvbin_size2, zorder=10)
            '''

            ax3.axhline(0, c='k', ls='--', zorder=10)
            ax8.axhline(1, c='k', ls='--', label='SNR = 1', zorder=10)
            eff_res = 1.5e6 #1.93e6 #2.45e6 #2.45e6
            ax5.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
            ax2.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
            ax10.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
            ax8.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
            ax5.text(.725, .1, '  Effective\nresolution', transform=ax5.transAxes, fontsize=20, color='#FF8B00')

            ax3.legend(fontsize=20, ncol=1)
            ax2.legend(fontsize=20, ncol=1)
            ax8.legend(fontsize=20)
            ax10.legend(fontsize=20)
            ax5.legend(fontsize=20)

            ax3.set_xlim(xlo1, xhi1)
            ax3.set_ylim(-.12,.12)
            ax5.set_ylim(-.08,.08)
            ax2.set_ylim(1.1e-7,.9)
            xlo, xhi = [9e3, 2.5e6] #ax5.get_xlim()
            ax5.set_xlim(xlo, xhi)
            ax2.set_xlim(xlo, xhi)
            ax10.set_xlim(xlo, xhi)
            ax8.set_xlim(xlo, xhi)
            ax8.set_ylim(1e-2,5)
            ax5.set_xscale('log')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax8.set_xscale('log')
            ax8.set_yscale('log')
            ax10.set_xscale('log')
            ax10.set_yscale('log')
            ax10.set_yticks([1e-4,1e-2,1e0,1e2])

            ax3.set_xlabel('r ["]', fontsize=24)
            ax3.set_ylabel('Normalized\nresidual', fontsize=24)
            ax5.set_ylabel('Re(V) [Jy]', fontsize=24)
            ax2.set_ylabel('Re(V) [Jy]', fontsize=24)
            ax10.set_ylabel('Norm. resid.', fontsize=24)
            ax8.set_ylabel('SNR', fontsize=24)
            ax8.set_xlabel(r'Baseline [$\lambda$]', fontsize=24)
            ax3.text(.25, .8, 'd)', transform=ax3.transAxes, fontsize=24)
            ax5.text(.75, .82, 'e)', transform=ax5.transAxes, fontsize=24)
            ax2.text(.75, .875, 'f)', transform=ax2.transAxes, fontsize=24)
            ax10.text(.5, .82, 'g)', transform=ax10.transAxes, fontsize=24)
            ax8.text(.1, .82, 'h)', transform=ax8.transAxes, fontsize=24)
            plt.setp(ax5.get_xticklabels(), visible=False)
            plt.setp(ax2.get_xticklabels(), visible=False)
            plt.setp(ax10.get_xticklabels(), visible=False)

        """


        #ax8.text(.1, .1, 'f)', transform=ax8.transAxes, fontsize=24)

        #ax9.set_ylim(1e-3, 1e1)

        #ax10.semilogx(uvbins, resid_vis, '.', c='#33D9FF')


        """
        '''
        ax4 = fig1.add_subplot(gs1_single[5])
        resid = binned_re - GPHF.HankelTransform(uvbins)
        rmse = (np.mean(resid ** 2)) ** .5
        ax4.semilogx(uvbins, (binned_re - GPHF.HankelTransform(uvbins)) / max(binned_re), label='RMSE %.2d'% (rmse / max(binned_re)))
        ax4.set_ylabel('Normalized residual', fontsize=24)
        xlo, xhi = ax4.get_xlim()
        ax4.set_xlim(xlo, xhi)
        #plt.legend(fontsize=10)
        plt.setp(ax4.get_xticklabels(), visible=False)
        '''
        '''
        ax4 = fig1.add_subplot(gs1_double[9])
        plt.ylabel('Power', fontsize=24)

        #gs_cbar = GridSpec(2, 8, right=.99)
        plt.loglog(uv_pts_toplot[0], pi_toplot[0] * 2 * np.pi, c=cmaps[counter_gp](0))#, label=r'%s bins, $\alpha$ =%s, ss=%.0e, %s iterations'%(nbins, alpha, smooth_strength, smooth_iterations))
        for l in range(len(uv_pts_toplot)):
            plt.loglog(uv_pts_toplot[l], pi_toplot[l] * 2 * np.pi, c=cmaps[counter_gp](l / len(uv_pts_toplot)))
        plt.xlim(xlo, xhi)

        plt.xlim(xlo, 1e7)

        #plt.legend(fontsize=10)
        plt.setp(ax4.get_xticklabels(), visible=False)
        '''
        '''
        ax6 = fig1.add_subplot(gs1_small[12])
        plt.ylabel(r'$\chi^2$', fontsize=24)

        #plt.plot(np.array(chi2s_alt) / len(uv.u), '.', label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        plt.plot(np.array(chi2s) / len(uv.u), label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        #plt.legend(fontsize=10)

        plt.setp(ax6.get_xticklabels(), visible=False)


        ax9 = fig1.add_subplot(gs1_small[14])
        plt.xlabel('Fit iteration #', fontsize=24)
        plt.ylabel('Con.#(i$_{cov}$)', fontsize=24)

        plt.semilogy(condition_numbers, label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        plt.yticks(ax9.get_ylim(), ('%.1e'%ax9.get_ylim()[0], '%.1e'%ax9.get_ylim()[1]))
        '''
        #'''
        """
        #ax8 = fig1.add_subplot(gs1_single[7])
        #ax8 = fig1.add_subplot(gs1_single[5])
        #ax8 = fig1.add_subplot(gs1_double[9])
        ax8 = fig1.add_subplot(gs1_single2[3])
        #ax8.text(.95, .8, 'f)', transform=ax8.transAxes, fontsize=24)
        #ax8.text(.1, .1, 'f)', transform=ax8.transAxes, fontsize=24)
        #plt.ylabel(r'SNR = |Re(V)| $\cdot$ $\sqrt{\rm w}$', fontsize=24)
        plt.ylabel(r'SNR', fontsize=24)
        plt.xlabel(r'Baseline [$\lambda$]', fontsize=24)
        xlo, xhi = ax1.get_xlim()
        ax8.set_xlim(xlo, xhi)

        #ax8.set_xlim(xlo, 4e6)
        #ax8.set_xlim(xlo, 1e7)

        #ax8.set_ylim(1e-1, 1e1)
        #ax8.set_ylim(1e-3, 1e1)
        ax8.set_ylim(0, 5)
        print('start snr')
        if counter_gp == 0:
            #'''

            #plt.legend(fontsize=24, loc=[.5, .05])
        #'''
        """
        '''
        plt.axhline(0, c='#A4A4A4', ls='--')
        plt.xlabel('Baseline [$\lambda$]', fontsize=24)
        plt.ylabel('Im(V) [Jy]', fontsize=24)
        if counter_gp == 0:
            plt.errorbar(uvbins, binned_im, yerr=binned_im_err, marker='.', ls='none', c='k', label='obs, %.0f k$\lambda$ bins'%(uvbin_size/1e3))
        ax8.set_xscale('log')
        plt.xlim(xlo, xhi)
        plt.legend(fontsize=10)
        '''

        '''
        corr = cov_renorm.copy()
        for i in range(len(mu)):
            corr[i] /= np.sqrt(cov_renorm[i, i])
            corr[:, i] /= np.sqrt(cov_renorm[i, i])
        '''
        #"""
        #ax10.axvline(eff_res, zorder=20, c='r', ls='--', label='RMSE[:eff. res.] %.3f'%round(rmse_vis,3))
        #ax8.axvline(eff_res, zorder=20, c='b', ls='--')
        #ax5.legend(loc=[.4,.3])
        #ax10.legend(fontsize=24, ncol=1)
        #"""
        plt.savefig(plot_msfigs_savefile)#, dpi=600)
        #plt.show()


def ms_f2(plot_msfigs_savefile, disc, Rmax, baselines, uv, model, GPHF, mu, nbins, alpha, smooth_strength, uvbin_size, uvbin_size2, uv_pts_toplot, pi_toplot, enforce_positivity, smooth_iterations, cut, counter_gp, known_profile, image_extracted_profile, add_noise, nfits, cmaps, snr_plot_binwidth, snr_plot_binwidth2, plot_savefile, savedir):
        #plt.ion()
        import matplotlib as mpl
        mpl.rcParams['xtick.labelsize'] = 24
        mpl.rcParams['ytick.labelsize'] = 24

        #gs1_single = GridSpec(4, 2, bottom=.18, top=.98, left=.05, right=.97, hspace=0)
        gs1_single = GridSpec(2, 2, bottom=.1, top=.98, left=.09, right=.98, hspace=0)
        gs1_single2 = GridSpec(2, 2, bottom=.085, top=.7, left=.09, right=.98, hspace=0)
        gs1_single_offset = GridSpec(3, 2, bottom=-0.07, top=.83, left=.07, right=.98, hspace=0)
        gs1_big = GridSpec(1, 2, bottom=.25, top=.98, left=.09, right=.98, hspace=0)
        gs1_big2 = GridSpec(1, 2, bottom=.085, top=.75, left=.09, right=.98, hspace=0)
        gs1_big3 = GridSpec(1, 2, bottom=.30875, top=.75625, left=.09, right=.98, hspace=0)
        gs1_double = GridSpec(4, 2, bottom=.085, top=.98, left=.09, right=.98, hspace=0)
        gs1_double_top = GridSpec(4, 2, bottom=.185, top=.98, left=.09, right=.98, hspace=0)
        gs1_double_mid = GridSpec(4, 2, bottom=-.115, top=1.08, left=.09, right=.98, hspace=0)
        gs1_double_bot = GridSpec(4, 2, bottom=.085, top=.88, left=.09, right=.98, hspace=0)
        gs1_double2 = GridSpec(5, 2, bottom=.085, top=.98, left=.09, right=.98, hspace=0)
        gs1 = GridSpec(4, 2, bottom=.15, top=.98, left=.05, right=.97, hspace=0)
        gs1_2 = GridSpec(4, 2, bottom=.08, top=.89, left=.05, right=.97, hspace=0)
        gs1_small = GridSpec(8, 2, bottom=.08, top=.72, left=.05, right=.97, hspace=0)
        gs2 = GridSpec(4, 2)

        fig1 = plt.figure()

        grid = np.linspace(0, Rmax, 10000)
        ki = 2*np.pi*np.logspace(np.log10(min(baselines.min(), GPHF.uv_pts[0])),
                                 np.log10(max(baselines.max(), GPHF.uv_pts[-1])),
                                 10**4)

        ax1 = fig1.add_subplot(gs1_double[0])

        resid = model(GPHF.Rc[:cut]) - mu[:cut]
        rmse = (np.mean(resid ** 2)) ** .5

        clean_r, clean_i, _  = np.genfromtxt(pwd + '/../ALMAsim/' + disc + '/' + disc + '_cleaned_image_radial_profile.txt').T
        interp_func = interp1d(clean_r, clean_i, kind='nearest', bounds_error=False)
        regrid_clean_i = interp_func(GPHF.Rc * rad_to_arcsec)[:-1]

        resid_clean = model(GPHF.Rc[:cut]) - regrid_clean_i
        rmse_clean = (np.nanmean(resid_clean ** 2)) ** .5 # NaNs arise
        # when this profile doesn't extend to as large radii as the fit

        ax1.plot(grid * rad_to_arcsec, model(grid) / 1e10, 'k--', label='Input profile')#label='Andrews et al. 2018 (for reference)')
        ax1.plot(GPHF.Rc[:cut] * rad_to_arcsec, regrid_clean_i / 1e10, c='#a4a4a4', label=r'CLEAN, natural')
        ax1.plot(GPHF.Rc[:cut] * rad_to_arcsec, mu[:cut] / 1e10, c='r', label='Fit')#label=r'Fit to Fedele et al. 2018 data') #AD13FF

        #plt.errorbar(GPHF.Rc * rad_to_arcsec, mu, yerr=np.diag(cov_renorm)**0.5, fmt='+', label='bins', zorder=-1)
        #plt.fill_between(GPHF.Rc * rad_to_arcsec, mu - np.diag(cov_renorm)**0.5, mu + np.diag(cov_renorm)**0.5, where=  mu - np.diag(cov_renorm)**0.5 >=  mu + np.diag(cov_renorm)**0.5, facecolor='r', alpha=.1)

        if disc == 'gauss_ring_pt0125_briggs_negative2_rev3':
            np.save(savedir + 'fit_temp.npy', [grid * rad_to_arcsec, model(grid) / 1e10, GPHF.Rc[:cut] * rad_to_arcsec, regrid_clean_i / 1e10, mu[:cut] / 1e10, resid, resid_clean, rmse, rmse_clean])
        g1_grid, g1_mod, g1_grid2, g1_clean, g1_mu, g1_resid, g1_resid_clean, g1_rmse, g1_rmse_clean = np.load(savedir + '../gauss_ring_pt0125_briggs_negative2_rev3/fit_temp.npy').T
        ax0_1 = fig1.add_subplot(gs1_double[4])
        ax0_2 = fig1.add_subplot(gs1_double[6])
        ax0_1.plot(g1_grid, g1_mod, 'k--', label='Input profile')#label='Andrews et al. 2018 (for reference)')
        ax0_1.plot(g1_grid2, g1_clean, c='#a4a4a4', label=r'CLEAN, natural')
        ax0_1.plot(g1_grid2, g1_mu, c='r', label='Fit')#label=r'Fit to Fedele et al. 2018 data') #AD13FF
        ax0_2.plot(g1_grid2, g1_resid_clean / 1e10 / max(g1_mod), '.', c='#a4a4a4',
             label='CLEAN, RMSE %.3f'% round(g1_rmse_clean / 1e10 / max(g1_mod),3))
        ax0_2.plot(g1_grid2, g1_resid / 1e10 / max(g1_mod), '.', c='r', label=r'Fit, RMSE %.3f' % round(g1_rmse / 1e10 / max(g1_mod),3))
        #plt.setp(ax0_2.get_xticklabels(), visible=False)

        xlo1, xhi1 = ax1.get_xlim()

        ax1.legend(fontsize=20, ncol=1, loc='best')
        ax1.set_ylabel('Brightness\n[$10^{10}$ Jy sr$^{-1}$]', fontsize=24)
        plt.setp(ax1.get_xticklabels(), visible=False)

        ax1.text(.1, .8, 'a)', transform=ax1.transAxes, fontsize=24)

        ax3 = fig1.add_subplot(gs1_double[2])
        ax5 = fig1.add_subplot(gs1_double_top[1])
        ax2 = fig1.add_subplot(gs1_double_mid[3])
        ax10 = fig1.add_subplot(gs1_double_bot[5])
        ax8 = fig1.add_subplot(gs1_double_bot[7])

        uvbins, binned_re, binned_im, binned_re_err, binned_im_err = \
            uv.plot(linestyle='+', color='#A4A4A4',
                label=r'Obs, %.0f k$\lambda$ bins' % (uvbin_size),
                uvbin_size=uvbin_size, fontsize=24, return_binned_values=True)

        uvbins2, binned_re2, binned_im2, binned_re_err2, binned_im_err2 = \
            uv.plot(linestyle='x', linestyle2='+', color='#A4A4A4',
                label=r'obs, %.0f k$\lambda$ bins' % (uvbin_size2),
                uvbin_size=uvbin_size2, fontsize=24, return_binned_values=True)

        def find_nearest(array,value):
        	idx = (np.abs(array - value)).argmin()
        	#print 'nearest value %s. array position %s'%(array[idx],idx)
        	return idx

        snr_barrier = 5e5
        resid_vis = binned_re - GPHF.HankelTransform(uvbins * 2 * np.pi)
        norm_resid_vis = (binned_re - GPHF.HankelTransform(uvbins * 2 * np.pi)) / binned_re
        print('find_nearest',find_nearest(uvbins, snr_barrier))
        resid_vis_to_barrier = resid_vis[:find_nearest(uvbins, snr_barrier)]
        rmse_vis = (np.mean(resid_vis_to_barrier ** 2)) ** .5
        norm_resid_vis_to_barrier = norm_resid_vis[:find_nearest(uvbins, snr_barrier)]
        norm_rmse_vis = (np.mean(norm_resid_vis_to_barrier ** 2)) ** .5
        #print('norm_resid_vis',norm_resid_vis)

        sortidx = np.argsort(baselines)
        bls = baselines[sortidx] / 1e3
        snr = abs(uv.re[sortidx]) / (1 / uv.weights[sortidx] ** .5)
        binwidth = snr_plot_binwidth / 1e3

        bins = np.arange(0, max(bls) + binwidth, binwidth)
        mids = bins[:-1] + binwidth / 2
        bins_snr = np.vstack((np.digitize(bls, bins), snr)).T
        mean_snr = [np.mean(bins_snr[bins_snr[:, 0] == i, 1]) for i in
                    range(len(bins))]  # Runtimewarning here; how to ignore?
        std_snr = [np.std(bins_snr[bins_snr[:, 0] == i, 1]) for i in range(len(bins))]
        ax8.errorbar(mids * 1e3, mean_snr[1:], yerr=std_snr[1:], color='#a4a4a4',fmt='.', ecolor='k',
                     label='%.0f k$\lambda$ bins' % binwidth)

        binwidth = snr_plot_binwidth2 / 1e3
        bins = np.arange(0, max(bls) + binwidth, binwidth)
        mids = bins[:-1] + binwidth / 2
        bins_snr = np.vstack((np.digitize(bls, bins), snr)).T
        mean_snr = [np.mean(bins_snr[bins_snr[:, 0] == i, 1]) for i in
                    range(len(bins))]  # Runtimewarning here; how to ignore?
        std_snr = [np.std(bins_snr[bins_snr[:, 0] == i, 1]) for i in range(len(bins))]
        print('NB the bin centering for SNR plot is wrong; should use the results from uvplot')
        ax8.errorbar(mids * 1e3, mean_snr[1:], yerr=std_snr[1:], color='m',fmt='.', ecolor='#00AAFF', alpha=.5,
                     label='%.0f k$\lambda$' % binwidth, zorder=10)

        ax3.plot(GPHF.Rc[:cut] * rad_to_arcsec, resid_clean / max(model(GPHF.Rc[:cut])), '.', c='#a4a4a4',
             label='CLEAN, RMSE %.3f'% round(rmse_clean / max(model(GPHF.Rc[:cut])),3))
        ax3.plot(GPHF.Rc[:cut] * rad_to_arcsec, resid[:cut] / max(model(GPHF.Rc[:cut])), '.', c='r', label=r'Fit, RMSE %.3f' % round(rmse / max(model(GPHF.Rc[:cut])),3))

        ax5.plot(uvbins, binned_re, '+', c='c', ms=8, label=r'Obs., %.0f k$\lambda$ bins'%(uvbin_size/1e3))
        ax2.plot(baselines, uv.re, marker='.', ls='none', c='k', label='Obs. > 0') #k
        ax2.plot(baselines, -uv.re, marker='.', ls='none', c='#A4A4A4', label='< 0')
        # plt.errorbar(baselines, uv.re, yerr=uv.weights**-0.5,marker='+', ls='none', c='k', label='obs')
        # plt.errorbar(baselines, -uv.re, yerr=uv.weights**-0.5,marker='x', ls='none', c='c')
        ax2.plot(uvbins, binned_re, marker='+', ms=8, ls='none', color='c', label=r'> 0, %.0f k$\lambda$ bins'%(uvbin_size/1e3)) #33D9FF
        ax2.plot(uvbins, -binned_re, marker='+', ms=8, ls='none', color='#65EB2F', label=r'< 0, %.0f k$\lambda$'%(uvbin_size/1e3)) #9CE73C

        #ax2.errorbar(uvbins, binned_re, yerr=binned_re_err, marker='+', color='r', label=r'Obs. > 0, %.0f k$\lambda$ bins'%(uvbin_size * 50 / 1e3))
        #ax2.errorbar(uvbins, -binned_re, yerr=binned_re_err, marker='x', color='m', label=r'Obs. < 0, %.0f k$\lambda$ bins'%(uvbin_size * 50 / 1e3))
        #plt.errorbar(ki / (2 * np.pi), GPHF.HankelTransform(ki), yerr=np.diag(cov_renorm)**0.5, fmt='+', label='bins', zorder=-1)
        ax5.plot(uvbins2, binned_re2, 'x', c='b', ms=8, label=r'%.0f k$\lambda$'%(uvbin_size2 / 1e3))
        ax5.plot(ki / (2 * np.pi), GPHF.HankelTransform(ki), c='r')#c='#AD13FF')#, label=r'%s bins, $\alpha$ =%s, ss=%.0e' % (nbins, alpha, smooth_strength))
        ax2.plot(uvbins2, binned_re2, marker='x', ms=8, ls='none', c='b', label=r'> 0, %.0f k$\lambda$'%(uvbin_size2 / 1e3))
        ax2.plot(uvbins2, -binned_re2, marker='x', ms=8, ls='none', c='m', label=r'< 0, %.0f k$\lambda$'%(uvbin_size2 / 1e3)) #880901
        plot_log_abs(ki/(2*np.pi), GPHF.HankelTransform(ki), ax2, c='r')#c='#AD13FF')#, alpha=.5)#, label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins,alpha,smooth_strength))

        #ax10.plot(uvbins, norm_resid_vis, '+', ms=8, c='#F700FF', label='> 0, %.0f k$\lambda$'%(uvbin_size/1e3))
        #ax10.plot(uvbins, -norm_resid_vis, 'x',  ms=8, c='#00FFAA', alpha=.5, label='< 0, %.0f k$\lambda$'%(uvbin_size/1e3))
        #gs_cbar = GridSpec(2, 8, right=.99)
        ax10.plot(uv_pts_toplot[0], pi_toplot[0] * 2 * np.pi, c=cmaps[counter_gp](0))#, label=r'%s bins, $\alpha$ =%s, ss=%.0e, %s iterations'%(nbins, alpha, smooth_strength, smooth_iterations))
        for l in range(len(uv_pts_toplot)):
            ax10.plot(uv_pts_toplot[l], pi_toplot[l] * 2 * np.pi, c=cmaps[counter_gp](l / len(uv_pts_toplot)))

        '''
        ax8.errorbar(uvbins, binned_re, yerr=binned_re_err, color='#00FF6C',fmt='.', ecolor='#BDBDBD', alpha=.5,
                     label='%.0f k$\lambda$' % uvbin_size, zorder=10)
        ax8.errorbar(uvbins2, binned_re2, yerr=binned_re_err2, color='r',fmt='.', ecolor='#00AAFF', alpha=.5,
                     label='%.0f k$\lambda$' % uvbin_size2, zorder=10)
        '''

        ax3.axhline(0, c='k', ls='--', zorder=10)
        ax8.axhline(1, c='k', ls='--', label='SNR = 1', zorder=10)
        eff_res = 1.5e6 #1.93e6 #2.45e6 #2.45e6
        #ax5.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
        #ax2.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
        #ax10.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
        #ax8.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
        #ax5.text(.725, .1, '  Effective\nresolution', transform=ax5.transAxes, fontsize=20, color='#FF8B00')

        ax0_1.legend(fontsize=20)
        ax0_2.legend(fontsize=20, loc='lower right')
        ax3.legend(fontsize=20)
        ax2.legend(fontsize=20, ncol=1, loc=[.0001,.015])
        ax8.legend(fontsize=20, loc=[.005,.015])
        ax10.legend(fontsize=20)
        ax5.legend(fontsize=20)

        ax0_2.set_ylim(-.8,.8)
        ax0_1.set_xlim(xlo1, xhi1)
        ax0_2.set_xlim(xlo1, xhi1)
        ax3.set_xlim(xlo1, xhi1)
        ax3.set_ylim(-.65,.65)
        #ax5.set_ylim(-.08,.08)
        ax2.set_ylim(bottom=1.1e-7)#,.9)
        xlo, xhi = [9e3, 2.5e6] #ax5.get_xlim()
        ax5.set_xlim(xlo, 4e6)
        ax2.set_xlim(xlo, 4e6)
        ax10.set_xlim(xlo, 4e6)
        ax8.set_xlim(xlo, 4e6)
        ax8.set_ylim(1e-2,10)
        ax5.set_xscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax8.set_xscale('log')
        ax8.set_yscale('log')
        ax10.set_xscale('log')
        ax10.set_yscale('log')
        ax10.set_yticks([1e-9,1e-7,1e-5])
        ax8.set_yticks([1e-2,1e-1,1e0,1e1])

        ax0_1.set_ylabel('Brightness\n[$10^{10}$ Jy sr$^{-1}$]', fontsize=24)
        ax0_2.set_ylabel('Normalized\nresidual', fontsize=24)
        #ax3.set_xlabel('r ["]', fontsize=24)
        ax3.set_ylabel('Normalized\nresidual', fontsize=24)
        ax5.set_ylabel('Re(V) [Jy]', fontsize=24)
        ax2.set_ylabel('Re(V) [Jy]', fontsize=24)
        #ax10.set_ylabel('Norm. resid.', fontsize=24)
        ax10.set_ylabel('Power', fontsize=24)
        ax8.set_ylabel('SNR', fontsize=24)
        ax8.set_xlabel(r'Baseline [$\lambda$]', fontsize=24)
        ax3.text(.1, .8, 'b)', transform=ax3.transAxes, fontsize=24)
        ax5.text(.9, .82, 'e)', transform=ax5.transAxes, fontsize=24)
        ax2.text(.9, .82, 'f)', transform=ax2.transAxes, fontsize=24)
        ax10.text(.3, .1, 'g)', transform=ax10.transAxes, fontsize=24)
        ax8.text(.3, .1, 'h)', transform=ax8.transAxes, fontsize=24)
        ax0_1.text(.1, .8, 'c)', transform=ax0_1.transAxes, fontsize=24)
        ax0_2.text(.1, .8, 'd)', transform=ax0_2.transAxes, fontsize=24)
        plt.setp(ax5.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax10.get_xticklabels(), visible=False)
        plt.setp(ax3.get_xticklabels(), visible=False)
        plt.setp(ax0_1.get_xticklabels(), visible=False)

        '''
        ax4 = fig1.add_subplot(gs1_single[5])
        resid = binned_re - GPHF.HankelTransform(uvbins)
        rmse = (np.mean(resid ** 2)) ** .5
        ax4.semilogx(uvbins, (binned_re - GPHF.HankelTransform(uvbins)) / max(binned_re), label='RMSE %.2d'% (rmse / max(binned_re)))
        ax4.set_ylabel('Normalized residual', fontsize=24)
        xlo, xhi = ax4.get_xlim()
        ax4.set_xlim(xlo, xhi)
        #plt.legend(fontsize=10)
        plt.setp(ax4.get_xticklabels(), visible=False)
        '''
        '''

        '''
        '''
        ax6 = fig1.add_subplot(gs1_small[12])
        plt.ylabel(r'$\chi^2$', fontsize=24)

        #plt.plot(np.array(chi2s_alt) / len(uv.u), '.', label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        plt.plot(np.array(chi2s) / len(uv.u), label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        #plt.legend(fontsize=10)

        plt.setp(ax6.get_xticklabels(), visible=False)


        ax9 = fig1.add_subplot(gs1_small[14])
        plt.xlabel('Fit iteration #', fontsize=24)
        plt.ylabel('Con.#(i$_{cov}$)', fontsize=24)

        plt.semilogy(condition_numbers, label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        plt.yticks(ax9.get_ylim(), ('%.1e'%ax9.get_ylim()[0], '%.1e'%ax9.get_ylim()[1]))
        '''
        #'''
        """
        #ax8 = fig1.add_subplot(gs1_single[7])
        #ax8 = fig1.add_subplot(gs1_single[5])
        #ax8 = fig1.add_subplot(gs1_double[9])
        ax8 = fig1.add_subplot(gs1_single2[3])
        #ax8.text(.95, .8, 'f)', transform=ax8.transAxes, fontsize=24)
        #ax8.text(.1, .1, 'f)', transform=ax8.transAxes, fontsize=24)
        #plt.ylabel(r'SNR = |Re(V)| $\cdot$ $\sqrt{\rm w}$', fontsize=24)
        plt.ylabel(r'SNR', fontsize=24)
        plt.xlabel(r'Baseline [$\lambda$]', fontsize=24)
        xlo, xhi = ax1.get_xlim()
        ax8.set_xlim(xlo, xhi)

        #ax8.set_xlim(xlo, 4e6)
        #ax8.set_xlim(xlo, 1e7)

        #ax8.set_ylim(1e-1, 1e1)
        #ax8.set_ylim(1e-3, 1e1)
        ax8.set_ylim(0, 5)
        print('start snr')
        if counter_gp == 0:
            #'''

            #plt.legend(fontsize=24, loc=[.5, .05])
        #'''
        """
        '''
        plt.axhline(0, c='#A4A4A4', ls='--')
        plt.xlabel('Baseline [$\lambda$]', fontsize=24)
        plt.ylabel('Im(V) [Jy]', fontsize=24)
        if counter_gp == 0:
            plt.errorbar(uvbins, binned_im, yerr=binned_im_err, marker='.', ls='none', c='k', label='obs, %.0f k$\lambda$ bins'%(uvbin_size/1e3))
        ax8.set_xscale('log')
        plt.xlim(xlo, xhi)
        plt.legend(fontsize=10)
        '''

        '''
        corr = cov_renorm.copy()
        for i in range(len(mu)):
            corr[i] /= np.sqrt(cov_renorm[i, i])
            corr[:, i] /= np.sqrt(cov_renorm[i, i])
        '''
        #"""
        #ax10.axvline(eff_res, zorder=20, c='r', ls='--', label='RMSE[:eff. res.] %.3f'%round(rmse_vis,3))
        #ax8.axvline(eff_res, zorder=20, c='b', ls='--')
        #ax5.legend(loc=[.4,.3])
        #ax10.legend(fontsize=24, ncol=1)
        #"""
        plt.savefig(plot_msfigs_savefile)#, dpi=600)
        #plt.show()


def ms_f3(plot_msfigs_savefile, disc, Rmax, baselines, uv, model, GPHF, mu, nbins, alpha, smooth_strength, uvbin_size, uvbin_size2, uv_pts_toplot, pi_toplot, enforce_positivity, smooth_iterations, cut, counter_gp, known_profile, image_extracted_profile, add_noise, nfits, cmaps, snr_plot_binwidth, snr_plot_binwidth2, plot_savefile, savedir):
        #plt.ion()
        import matplotlib as mpl
        mpl.rcParams['xtick.labelsize'] = 24
        mpl.rcParams['ytick.labelsize'] = 24

        #gs1_single = GridSpec(4, 2, bottom=.18, top=.98, left=.05, right=.97, hspace=0)
        gs1_single = GridSpec(2, 2, bottom=.1, top=.98, left=.1, right=.98, hspace=0)
        gs1_single2 = GridSpec(2, 2, bottom=.085, top=.7, left=.1, right=.98, hspace=0)
        gs1_single_offset = GridSpec(3, 2, bottom=-0.07, top=.83, left=.07, right=.98, hspace=0)
        gs1_big = GridSpec(1, 2, bottom=.25, top=.98, left=.1, right=.98, hspace=0)
        gs1_big2 = GridSpec(1, 2, bottom=.085, top=.75, left=.1, right=.98, hspace=0)
        gs1_big3 = GridSpec(1, 2, bottom=.30875, top=.75625, left=.1, right=.98, hspace=0)
        gs1_double = GridSpec(4, 2, bottom=.085, top=.98, left=.1, right=.98, hspace=0)
        gs1_double_top = GridSpec(4, 2, bottom=.185, top=.98, left=.1, right=.98, hspace=0)
        gs1_double_mid = GridSpec(4, 2, bottom=-.115, top=1.08, left=.1, right=.98, hspace=0)
        gs1_double_bot = GridSpec(4, 2, bottom=.085, top=.88, left=.1, right=.98, hspace=0)
        gs1_double2 = GridSpec(5, 2, bottom=.085, top=.98, left=.1, right=.98, hspace=0)
        gs1 = GridSpec(4, 2, bottom=.15, top=.98, left=.05, right=.97, hspace=0)
        gs1_2 = GridSpec(4, 2, bottom=.08, top=.89, left=.05, right=.97, hspace=0)
        gs1_small = GridSpec(8, 2, bottom=.08, top=.72, left=.05, right=.97, hspace=0)
        gs2 = GridSpec(4, 2)

        fig1 = plt.figure()

        grid = np.linspace(0, Rmax, 10000)
        ki = 2*np.pi*np.logspace(np.log10(min(baselines.min(), GPHF.uv_pts[0])),
                                 np.log10(max(baselines.max(), GPHF.uv_pts[-1])),
                                 10**4)

        ax1 = fig1.add_subplot(gs1_double[0])

        resid = model(GPHF.Rc[:cut]) - mu[:cut]
        rmse = (np.mean(resid ** 2)) ** .5

        clean_r, clean_i, _  = np.genfromtxt(pwd + '/../ALMAsim/' + disc + '/' + disc + '_cleaned_image_radial_profile.txt').T
        interp_func = interp1d(clean_r, clean_i, kind='nearest', bounds_error=False)
        regrid_clean_i = interp_func(GPHF.Rc * rad_to_arcsec)[:-1]

        resid_clean = model(GPHF.Rc[:cut]) - regrid_clean_i
        rmse_clean = (np.nanmean(resid_clean ** 2)) ** .5 # NaNs arise
        # when this profile doesn't extend to as large radii as the fit

        ax1.plot(grid * rad_to_arcsec, model(grid) / 1e10, 'k--', label='Input profile')#label='Andrews et al. 2018 (for reference)')
        #ax1.plot(GPHF.Rc[:cut] * rad_to_arcsec, regrid_clean_i / 1e10, c='#a4a4a4', label=r'CLEAN, Briggs 0.5')
        ax1.plot(GPHF.Rc[:cut] * rad_to_arcsec, mu[:cut] / 1e10, c='r', label='Fit')#label=r'Fit to Fedele et al. 2018 data') #AD13FF

        #plt.errorbar(GPHF.Rc * rad_to_arcsec, mu, yerr=np.diag(cov_renorm)**0.5, fmt='+', label='bins', zorder=-1)
        #plt.fill_between(GPHF.Rc * rad_to_arcsec, mu - np.diag(cov_renorm)**0.5, mu + np.diag(cov_renorm)**0.5, where=  mu - np.diag(cov_renorm)**0.5 >=  mu + np.diag(cov_renorm)**0.5, facecolor='r', alpha=.1)

        xlo1, xhi1 = ax1.get_xlim()

        ax3 = fig1.add_subplot(gs1_double[2])
        ax5 = fig1.add_subplot(gs1_double_top[1])
        ax2 = fig1.add_subplot(gs1_double_mid[3])
        ax10 = fig1.add_subplot(gs1_double_bot[5])
        ax8 = fig1.add_subplot(gs1_double_bot[7])

        uvbins, binned_re, binned_im, binned_re_err, binned_im_err = \
            uv.plot(linestyle='+', color='#A4A4A4',
                label=r'Obs, %.0f k$\lambda$ bins' % (uvbin_size),
                uvbin_size=uvbin_size, fontsize=24, return_binned_values=True)

        uvbins2, binned_re2, binned_im2, binned_re_err2, binned_im_err2 = \
            uv.plot(linestyle='x', linestyle2='+', color='#A4A4A4',
                label=r'obs, %.0f k$\lambda$ bins' % (uvbin_size2),
                uvbin_size=uvbin_size2, fontsize=24, return_binned_values=True)

        def find_nearest(array,value):
        	idx = (np.abs(array - value)).argmin()
        	#print 'nearest value %s. array position %s'%(array[idx],idx)
        	return idx

        snr_barrier = 5e5
        resid_vis = binned_re - GPHF.HankelTransform(uvbins * 2 * np.pi)
        norm_resid_vis = (binned_re - GPHF.HankelTransform(uvbins * 2 * np.pi)) / binned_re
        print('find_nearest',find_nearest(uvbins, snr_barrier))
        resid_vis_to_barrier = resid_vis[:find_nearest(uvbins, snr_barrier)]
        rmse_vis = (np.mean(resid_vis_to_barrier ** 2)) ** .5
        norm_resid_vis_to_barrier = norm_resid_vis[:find_nearest(uvbins, snr_barrier)]
        norm_rmse_vis = (np.mean(norm_resid_vis_to_barrier ** 2)) ** .5
        #print('norm_resid_vis',norm_resid_vis)

        sortidx = np.argsort(baselines)
        bls = baselines[sortidx] / 1e3
        snr = abs(uv.re[sortidx]) / (1 / uv.weights[sortidx] ** .5)
        binwidth = snr_plot_binwidth / 1e3

        bins = np.arange(0, max(bls) + binwidth, binwidth)
        mids = bins[:-1] + binwidth / 2
        bins_snr = np.vstack((np.digitize(bls, bins), snr)).T
        mean_snr = [np.mean(bins_snr[bins_snr[:, 0] == i, 1]) for i in
                    range(len(bins))]  # Runtimewarning here; how to ignore?
        std_snr = [np.std(bins_snr[bins_snr[:, 0] == i, 1]) for i in range(len(bins))]
        ax8.errorbar(mids * 1e3, mean_snr[1:], yerr=std_snr[1:], color='#a4a4a4',fmt='.', ecolor='k',
                     label='%.0f k$\lambda$ bins' % binwidth)

        binwidth = snr_plot_binwidth2 / 1e3
        bins = np.arange(0, max(bls) + binwidth, binwidth)
        mids = bins[:-1] + binwidth / 2
        bins_snr = np.vstack((np.digitize(bls, bins), snr)).T
        mean_snr = [np.mean(bins_snr[bins_snr[:, 0] == i, 1]) for i in
                    range(len(bins))]  # Runtimewarning here; how to ignore?
        std_snr = [np.std(bins_snr[bins_snr[:, 0] == i, 1]) for i in range(len(bins))]
        print('NB the bin centering for SNR plot is wrong; should use the results from uvplot')
        ax8.errorbar(mids * 1e3, mean_snr[1:], yerr=std_snr[1:], color='m',fmt='.', ecolor='#00AAFF', alpha=.5,
                     label='%.0f k$\lambda$' % binwidth, zorder=10)

        #ax3.plot(GPHF.Rc[:cut] * rad_to_arcsec, resid_clean / max(model(GPHF.Rc[:cut])), '.', c='#a4a4a4',
        #     label='CLEAN, RMSE %.3f'% round(rmse_clean / max(model(GPHF.Rc[:cut])),3))
        ax3.plot(GPHF.Rc[:cut] * rad_to_arcsec, resid[:cut] / max(model(GPHF.Rc[:cut])), '.', c='r', label=r'RMSE %.3f' % round(rmse / max(model(GPHF.Rc[:cut])),3))

        ax5.plot(uvbins, binned_re, '+', c='c', ms=8, label=r'Obs., %.0f k$\lambda$ bins'%(uvbin_size/1e3))
        ax2.plot(baselines, uv.re, marker='.', ls='none', c='k', label='Obs. > 0') #k
        ax2.plot(baselines, -uv.re, marker='.', ls='none', c='#A4A4A4', label='< 0')
        # plt.errorbar(baselines, uv.re, yerr=uv.weights**-0.5,marker='+', ls='none', c='k', label='obs')
        # plt.errorbar(baselines, -uv.re, yerr=uv.weights**-0.5,marker='x', ls='none', c='c')
        ax2.plot(uvbins, binned_re, marker='+', ms=8, ls='none', color='c', label=r'> 0, %.0f k$\lambda$ bins'%(uvbin_size/1e3)) #33D9FF
        ax2.plot(uvbins, -binned_re, marker='+', ms=8, ls='none', color='#65EB2F', label=r'< 0, %.0f k$\lambda$'%(uvbin_size/1e3)) #9CE73C

        #ax2.errorbar(uvbins, binned_re, yerr=binned_re_err, marker='+', color='r', label=r'Obs. > 0, %.0f k$\lambda$ bins'%(uvbin_size * 50 / 1e3))
        #ax2.errorbar(uvbins, -binned_re, yerr=binned_re_err, marker='x', color='m', label=r'Obs. < 0, %.0f k$\lambda$ bins'%(uvbin_size * 50 / 1e3))
        #plt.errorbar(ki / (2 * np.pi), GPHF.HankelTransform(ki), yerr=np.diag(cov_renorm)**0.5, fmt='+', label='bins', zorder=-1)
        ax5.plot(uvbins2, binned_re2, 'x', c='b', ms=8, label=r'%.0f k$\lambda$'%(uvbin_size2 / 1e3))
        ax5.plot(ki / (2 * np.pi), GPHF.HankelTransform(ki), c='r')#c='#AD13FF')#, label=r'%s bins, $\alpha$ =%s, ss=%.0e' % (nbins, alpha, smooth_strength))
        ax2.plot(uvbins2, binned_re2, marker='x', ms=8, ls='none', c='b', label=r'> 0, %.0f k$\lambda$'%(uvbin_size2 / 1e3))
        ax2.plot(uvbins2, -binned_re2, marker='x', ms=8, ls='none', c='m', label=r'< 0, %.0f k$\lambda$'%(uvbin_size2 / 1e3)) #880901
        plot_log_abs(ki/(2*np.pi), GPHF.HankelTransform(ki), ax2, c='r')#c='#AD13FF')#, alpha=.5)#, label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins,alpha,smooth_strength))

        ax10.plot(uvbins, norm_resid_vis, '+', ms=8, c='#F700FF', label='> 0, %.0f k$\lambda$'%(uvbin_size/1e3))
        ax10.plot(uvbins, -norm_resid_vis, 'x',  ms=8, c='#00FFAA', alpha=.5, label='< 0, %.0f k$\lambda$'%(uvbin_size/1e3))

        '''
        ax8.errorbar(uvbins, binned_re, yerr=binned_re_err, color='#00FF6C',fmt='.', ecolor='#BDBDBD', alpha=.5,
                     label='%.0f k$\lambda$' % uvbin_size, zorder=10)
        ax8.errorbar(uvbins2, binned_re2, yerr=binned_re_err2, color='r',fmt='.', ecolor='#00AAFF', alpha=.5,
                     label='%.0f k$\lambda$' % uvbin_size2, zorder=10)
        '''

        ax3.axhline(0, c='k', ls='--', zorder=10)
        ax8.axhline(1, c='k', ls='--', label='SNR = 1', zorder=10)
        eff_res = 1.7e6 #1.93e6 #2.45e6 #2.45e6
        ax5.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
        ax2.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
        ax10.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
        ax8.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
        ax5.text(.725, .1, '  Effective\nresolution', transform=ax5.transAxes, fontsize=20, color='#FF8B00')

        ax1.legend(fontsize=20, ncol=1, loc='best')
        ax3.legend(fontsize=20, ncol=1)
        ax2.legend(fontsize=20, ncol=1)
        ax8.legend(fontsize=20)
        ax10.legend(fontsize=20)
        ax5.legend(fontsize=20)

        ax3.set_xlim(xlo1, xhi1)
        #ax3.set_ylim(-.12,.12)
        ax5.set_ylim(-.05,.05)
        ax2.set_ylim(1.1e-7,.9)
        xlo, xhi = [9e3, 2.5e6] #ax5.get_xlim()
        ax5.set_xlim(xlo, xhi)
        ax2.set_xlim(xlo, xhi)
        ax10.set_xlim(xlo, xhi)
        ax8.set_xlim(xlo, xhi)
        ax8.set_ylim(1e-2,5)
        ax5.set_xscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax8.set_xscale('log')
        ax8.set_yscale('log')
        ax10.set_xscale('log')
        ax10.set_yscale('log')
        #ax10.set_yticks([1e-4,1e-2,1e0,1e2])

        ax1.set_ylabel('Brightness\n[$10^{10}$ Jy sr$^{-1}$]', fontsize=24)
        ax3.set_xlabel('r ["]', fontsize=24)
        ax3.set_ylabel('Normalized\nresidual', fontsize=24)
        ax5.set_ylabel('Re(V) [Jy]', fontsize=24)
        ax2.set_ylabel('Re(V) [Jy]', fontsize=24)
        ax10.set_ylabel('Norm. resid.', fontsize=24)
        ax8.set_ylabel('SNR', fontsize=24)
        ax8.set_xlabel(r'Baseline [$\lambda$]', fontsize=24)
        ax1.text(.5, .8, 'a)', transform=ax1.transAxes, fontsize=24)
        ax3.text(.5, .8, 'b)', transform=ax3.transAxes, fontsize=24)
        ax5.text(.75, .82, 'c)', transform=ax5.transAxes, fontsize=24)
        ax2.text(.75, .875, 'd)', transform=ax2.transAxes, fontsize=24)
        ax10.text(.1, .1, 'e)', transform=ax10.transAxes, fontsize=24)
        ax8.text(.1, .82, 'f)', transform=ax8.transAxes, fontsize=24)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax5.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax10.get_xticklabels(), visible=False)

        """


        #ax8.text(.1, .1, 'f)', transform=ax8.transAxes, fontsize=24)

        #ax9.set_ylim(1e-3, 1e1)

        #ax10.semilogx(uvbins, resid_vis, '.', c='#33D9FF')


        """
        '''
        ax4 = fig1.add_subplot(gs1_single[5])
        resid = binned_re - GPHF.HankelTransform(uvbins)
        rmse = (np.mean(resid ** 2)) ** .5
        ax4.semilogx(uvbins, (binned_re - GPHF.HankelTransform(uvbins)) / max(binned_re), label='RMSE %.2d'% (rmse / max(binned_re)))
        ax4.set_ylabel('Normalized residual', fontsize=24)
        xlo, xhi = ax4.get_xlim()
        ax4.set_xlim(xlo, xhi)
        #plt.legend(fontsize=10)
        plt.setp(ax4.get_xticklabels(), visible=False)
        '''
        '''
        ax4 = fig1.add_subplot(gs1_double[9])
        plt.ylabel('Power', fontsize=24)

        #gs_cbar = GridSpec(2, 8, right=.99)
        plt.loglog(uv_pts_toplot[0], pi_toplot[0] * 2 * np.pi, c=cmaps[counter_gp](0))#, label=r'%s bins, $\alpha$ =%s, ss=%.0e, %s iterations'%(nbins, alpha, smooth_strength, smooth_iterations))
        for l in range(len(uv_pts_toplot)):
            plt.loglog(uv_pts_toplot[l], pi_toplot[l] * 2 * np.pi, c=cmaps[counter_gp](l / len(uv_pts_toplot)))
        plt.xlim(xlo, xhi)

        plt.xlim(xlo, 1e7)

        #plt.legend(fontsize=10)
        plt.setp(ax4.get_xticklabels(), visible=False)
        '''
        '''
        ax6 = fig1.add_subplot(gs1_small[12])
        plt.ylabel(r'$\chi^2$', fontsize=24)

        #plt.plot(np.array(chi2s_alt) / len(uv.u), '.', label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        plt.plot(np.array(chi2s) / len(uv.u), label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        #plt.legend(fontsize=10)

        plt.setp(ax6.get_xticklabels(), visible=False)


        ax9 = fig1.add_subplot(gs1_small[14])
        plt.xlabel('Fit iteration #', fontsize=24)
        plt.ylabel('Con.#(i$_{cov}$)', fontsize=24)

        plt.semilogy(condition_numbers, label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        plt.yticks(ax9.get_ylim(), ('%.1e'%ax9.get_ylim()[0], '%.1e'%ax9.get_ylim()[1]))
        '''
        #'''
        """
        #ax8 = fig1.add_subplot(gs1_single[7])
        #ax8 = fig1.add_subplot(gs1_single[5])
        #ax8 = fig1.add_subplot(gs1_double[9])
        ax8 = fig1.add_subplot(gs1_single2[3])
        #ax8.text(.95, .8, 'f)', transform=ax8.transAxes, fontsize=24)
        #ax8.text(.1, .1, 'f)', transform=ax8.transAxes, fontsize=24)
        #plt.ylabel(r'SNR = |Re(V)| $\cdot$ $\sqrt{\rm w}$', fontsize=24)
        plt.ylabel(r'SNR', fontsize=24)
        plt.xlabel(r'Baseline [$\lambda$]', fontsize=24)
        xlo, xhi = ax1.get_xlim()
        ax8.set_xlim(xlo, xhi)

        #ax8.set_xlim(xlo, 4e6)
        #ax8.set_xlim(xlo, 1e7)

        #ax8.set_ylim(1e-1, 1e1)
        #ax8.set_ylim(1e-3, 1e1)
        ax8.set_ylim(0, 5)
        print('start snr')
        if counter_gp == 0:
            #'''

            #plt.legend(fontsize=24, loc=[.5, .05])
        #'''
        """
        '''
        plt.axhline(0, c='#A4A4A4', ls='--')
        plt.xlabel('Baseline [$\lambda$]', fontsize=24)
        plt.ylabel('Im(V) [Jy]', fontsize=24)
        if counter_gp == 0:
            plt.errorbar(uvbins, binned_im, yerr=binned_im_err, marker='.', ls='none', c='k', label='obs, %.0f k$\lambda$ bins'%(uvbin_size/1e3))
        ax8.set_xscale('log')
        plt.xlim(xlo, xhi)
        plt.legend(fontsize=10)
        '''

        '''
        corr = cov_renorm.copy()
        for i in range(len(mu)):
            corr[i] /= np.sqrt(cov_renorm[i, i])
            corr[:, i] /= np.sqrt(cov_renorm[i, i])
        '''
        #"""
        #ax10.axvline(eff_res, zorder=20, c='r', ls='--', label='RMSE[:eff. res.] %.3f'%round(rmse_vis,3))
        #ax8.axvline(eff_res, zorder=20, c='b', ls='--')
        #ax5.legend(loc=[.4,.3])
        #ax10.legend(fontsize=24, ncol=1)
        #"""
        plt.savefig(plot_msfigs_savefile)#, dpi=600)
        #plt.show()


def ms_f4(plot_msfigs_savefile, plot_msfigs_savefile2, disc, Rmax, baselines, uv, model, GPHF, mu, nbins, alpha, smooth_strength, uvbin_size, uvbin_size2, uv_pts_toplot, pi_toplot, enforce_positivity, smooth_iterations, cut, counter_gp, known_profile, image_extracted_profile, add_noise, nfits, cmaps, snr_plot_binwidth, snr_plot_binwidth2, plot_savefile, savedir):
        #plt.ion()
        import matplotlib as mpl
        mpl.rcParams['xtick.labelsize'] = 24
        mpl.rcParams['ytick.labelsize'] = 24

        #gs1_single = GridSpec(4, 2, bottom=.18, top=.98, left=.05, right=.97, hspace=0)
        gs1_single = GridSpec(2, 2, bottom=.1, top=.98, left=.09, right=.99, hspace=0, wspace=.23)
        gs1_single2 = GridSpec(2, 2, bottom=.085, top=.98, left=.09, right=.99, hspace=0, wspace=.23)
        gs1_single_offset = GridSpec(3, 2, bottom=-0.07, top=.83, left=.08, right=.99, hspace=0, wspace=.23)
        gs1_big = GridSpec(1, 2, bottom=.25, top=.98, left=.09, right=.99, hspace=0, wspace=.23)
        gs1_big2 = GridSpec(1, 2, bottom=.085, top=.75, left=.09, right=.99, hspace=0, wspace=.23)
        gs1_big3 = GridSpec(1, 2, bottom=.30875, top=.75625, left=.09, right=.99, hspace=0, wspace=.23)
        gs1_double = GridSpec(4, 2, bottom=.085, top=.98, left=.09, right=.99, hspace=0, wspace=.23)
        gs1_double_top = GridSpec(4, 2, bottom=.185, top=.98, left=.09, right=.99, hspace=0, wspace=.23)
        gs1_double_mid = GridSpec(4, 2, bottom=-.115, top=1.08, left=.09, right=.99, hspace=0, wspace=.23)
        gs1_double_bot = GridSpec(4, 2, bottom=.085, top=.88, left=.09, right=.99, hspace=0, wspace=.23)
        gs1_double2 = GridSpec(5, 2, bottom=.085, top=.98, left=.09, right=.99, hspace=0, wspace=.23)
        gs1 = GridSpec(4, 2, bottom=.15, top=.98, left=.05, right=.97, hspace=0, wspace=.23)
        gs1_2 = GridSpec(4, 2, bottom=.08, top=.89, left=.05, right=.97, hspace=0, wspace=.23)
        gs1_small = GridSpec(8, 2, bottom=.08, top=.72, left=.05, right=.97, hspace=0, wspace=.23)
        gs2 = GridSpec(4, 2)

        fig1 = plt.figure()
        fig2 = plt.figure()

        grid = np.linspace(0, Rmax, 10000)
        ki = 2*np.pi*np.logspace(np.log10(min(baselines.min(), GPHF.uv_pts[0])),
                                 np.log10(max(baselines.max(), GPHF.uv_pts[-1])),
                                 10**4)

        ax1 = fig1.add_subplot(gs1_double[0])
        ax6 = fig1.add_subplot(gs1_double[2])

        resid = model(GPHF.Rc[:cut]) - mu[:cut]
        rmse = (np.mean(resid ** 2)) ** .5

        clean_r, clean_i, _  = np.genfromtxt(pwd + '/../ALMAsim/' + disc + '/' + disc + '_cleaned_image_radial_profile.txt').T
        interp_func = interp1d(clean_r, clean_i, kind='nearest', bounds_error=False)
        regrid_clean_i = interp_func(GPHF.Rc * rad_to_arcsec)[:-1]

        resid_clean = model(GPHF.Rc[:cut]) - regrid_clean_i
        rmse_clean = (np.nanmean(resid_clean ** 2)) ** .5 # NaNs arise
        # when this profile doesn't extend to as large radii as the fit

        ax1.plot(grid * rad_to_arcsec, model(grid) / 1e10, 'k--', label='Input profile')#label='Andrews et al. 2018 (for reference)')
        ax1.plot(GPHF.Rc[:cut] * rad_to_arcsec, regrid_clean_i / 1e10, c='#a4a4a4', label=r'CLEAN, Briggs 0.5')
        ax1.plot(GPHF.Rc[:cut] * rad_to_arcsec, mu[:cut] / 1e10, c='r', label='Fit')#label=r'Fit to Fedele et al. 2018 data') #AD13FF

        ax6.plot(grid * rad_to_arcsec, model(grid) / 1e10, 'k--', label='Input')#label='Andrews et al. 2018 (for reference)')
        ax6.plot(GPHF.Rc[:cut] * rad_to_arcsec, regrid_clean_i / 1e10, c='#a4a4a4', label=r'CLEAN')
        ax6.plot(GPHF.Rc[:cut] * rad_to_arcsec, mu[:cut] / 1e10, c='r', label='Fit')#label=r'Fit to Fedele et al. 2018 data') #AD13FF

        #plt.errorbar(GPHF.Rc * rad_to_arcsec, mu, yerr=np.diag(cov_renorm)**0.5, fmt='+', label='bins', zorder=-1)
        #plt.fill_between(GPHF.Rc * rad_to_arcsec, mu - np.diag(cov_renorm)**0.5, mu + np.diag(cov_renorm)**0.5, where=  mu - np.diag(cov_renorm)**0.5 >=  mu + np.diag(cov_renorm)**0.5, facecolor='r', alpha=.1)

        xlo1, xhi1 = ax1.get_xlim()

        #ax3 = fig1.add_subplot(gs1_double[2])
        ax3 = fig1.add_subplot(gs1_double[4])
        ax5 = fig1.add_subplot(gs1_single2[1])
        ax2 = fig1.add_subplot(gs1_single2[3])

        ax10 = fig2.add_subplot(gs1_double_bot[5])
        ax8 = fig2.add_subplot(gs1_double_bot[7])

        uvbins, binned_re, binned_im, binned_re_err, binned_im_err = \
            uv.plot(linestyle='+', color='#A4A4A4',
                label=r'Obs, %.0f k$\lambda$ bins' % (uvbin_size),
                uvbin_size=uvbin_size, fontsize=24, return_binned_values=True)

        uvbins2, binned_re2, binned_im2, binned_re_err2, binned_im_err2 = \
            uv.plot(linestyle='x', linestyle2='+', color='#A4A4A4',
                label=r'obs, %.0f k$\lambda$ bins' % (uvbin_size2),
                uvbin_size=uvbin_size2, fontsize=24, return_binned_values=True)

        def find_nearest(array,value):
        	idx = (np.abs(array - value)).argmin()
        	#print 'nearest value %s. array position %s'%(array[idx],idx)
        	return idx

        snr_barrier = 5e5
        resid_vis = binned_re - GPHF.HankelTransform(uvbins * 2 * np.pi)
        norm_resid_vis = (binned_re - GPHF.HankelTransform(uvbins * 2 * np.pi)) / binned_re
        print('find_nearest',find_nearest(uvbins, snr_barrier))
        resid_vis_to_barrier = resid_vis[:find_nearest(uvbins, snr_barrier)]
        rmse_vis = (np.mean(resid_vis_to_barrier ** 2)) ** .5
        norm_resid_vis_to_barrier = norm_resid_vis[:find_nearest(uvbins, snr_barrier)]
        norm_rmse_vis = (np.mean(norm_resid_vis_to_barrier ** 2)) ** .5
        #print('norm_resid_vis',norm_resid_vis)

        sortidx = np.argsort(baselines)
        bls = baselines[sortidx] / 1e3
        snr = abs(uv.re[sortidx]) / (1 / uv.weights[sortidx] ** .5)
        binwidth = snr_plot_binwidth / 1e3

        bins = np.arange(0, max(bls) + binwidth, binwidth)
        mids = bins[:-1] + binwidth / 2
        bins_snr = np.vstack((np.digitize(bls, bins), snr)).T
        mean_snr = [np.mean(bins_snr[bins_snr[:, 0] == i, 1]) for i in
                    range(len(bins))]  # Runtimewarning here; how to ignore?
        std_snr = [np.std(bins_snr[bins_snr[:, 0] == i, 1]) for i in range(len(bins))]
        ax8.errorbar(mids * 1e3, mean_snr[1:], yerr=std_snr[1:], color='#a4a4a4',fmt='.', ecolor='k',
                     label='%.0f k$\lambda$ bins' % binwidth)

        binwidth = snr_plot_binwidth2 / 1e3
        bins = np.arange(0, max(bls) + binwidth, binwidth)
        mids = bins[:-1] + binwidth / 2
        bins_snr = np.vstack((np.digitize(bls, bins), snr)).T
        mean_snr = [np.mean(bins_snr[bins_snr[:, 0] == i, 1]) for i in
                    range(len(bins))]  # Runtimewarning here; how to ignore?
        std_snr = [np.std(bins_snr[bins_snr[:, 0] == i, 1]) for i in range(len(bins))]
        print('NB the bin centering for SNR plot is wrong; should use the results from uvplot')
        ax8.errorbar(mids * 1e3, mean_snr[1:], yerr=std_snr[1:], color='m',fmt='.', ecolor='#00AAFF', alpha=.5,
                     label='%.0f k$\lambda$' % binwidth, zorder=10)

        ax3.plot(GPHF.Rc[:cut] * rad_to_arcsec, resid_clean / max(model(GPHF.Rc[:cut])), '.', c='#a4a4a4',
             label='CLEAN, RMSE %.3f'% round(rmse_clean / max(model(GPHF.Rc[:cut])),3))
        ax3.plot(GPHF.Rc[:cut] * rad_to_arcsec, resid[:cut] / max(model(GPHF.Rc[:cut])), '.', c='r', label=r'Fit, RMSE %.3f' % round(rmse / max(model(GPHF.Rc[:cut])),3))

        ax5.plot(uvbins, binned_re, '+', c='c', ms=8, label=r'Obs., %.0f k$\lambda$ bins'%(uvbin_size/1e3))
        ax2.plot(baselines, uv.re, marker='.', ls='none', c='k', label='Obs. > 0') #k
        ax2.plot(baselines, -uv.re, marker='.', ls='none', c='#A4A4A4', label='< 0')
        # plt.errorbar(baselines, uv.re, yerr=uv.weights**-0.5,marker='+', ls='none', c='k', label='obs')
        # plt.errorbar(baselines, -uv.re, yerr=uv.weights**-0.5,marker='x', ls='none', c='c')
        ax2.plot(uvbins, binned_re, marker='+', ms=8, ls='none', color='c', label=r'> 0, %.0f k$\lambda$ bins'%(uvbin_size/1e3)) #33D9FF
        ax2.plot(uvbins, -binned_re, marker='+', ms=8, ls='none', color='#65EB2F', label=r'< 0, %.0f k$\lambda$'%(uvbin_size/1e3)) #9CE73C

        #ax2.errorbar(uvbins, binned_re, yerr=binned_re_err, marker='+', color='r', label=r'Obs. > 0, %.0f k$\lambda$ bins'%(uvbin_size * 50 / 1e3))
        #ax2.errorbar(uvbins, -binned_re, yerr=binned_re_err, marker='x', color='m', label=r'Obs. < 0, %.0f k$\lambda$ bins'%(uvbin_size * 50 / 1e3))
        #plt.errorbar(ki / (2 * np.pi), GPHF.HankelTransform(ki), yerr=np.diag(cov_renorm)**0.5, fmt='+', label='bins', zorder=-1)
        ax5.plot(uvbins2, binned_re2, 'x', c='b', ms=8, label=r'%.0f k$\lambda$'%(uvbin_size2 / 1e3))
        ax5.plot(ki / (2 * np.pi), GPHF.HankelTransform(ki), c='r')#c='#AD13FF')#, label=r'%s bins, $\alpha$ =%s, ss=%.0e' % (nbins, alpha, smooth_strength))
        ax2.plot(uvbins2, binned_re2, marker='x', ms=8, ls='none', c='b', label=r'> 0, %.0f k$\lambda$'%(uvbin_size2 / 1e3))
        ax2.plot(uvbins2, -binned_re2, marker='x', ms=8, ls='none', c='m', label=r'< 0, %.0f k$\lambda$'%(uvbin_size2 / 1e3)) #880901
        plot_log_abs(ki/(2*np.pi), GPHF.HankelTransform(ki), ax2, c='r')#c='#AD13FF')#, alpha=.5)#, label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins,alpha,smooth_strength))

        ax10.plot(uvbins, norm_resid_vis, '+', ms=8, c='#F700FF', label='> 0, %.0f k$\lambda$'%(uvbin_size/1e3))
        ax10.plot(uvbins, -norm_resid_vis, 'x',  ms=8, c='#00FFAA', alpha=.5, label='< 0, %.0f k$\lambda$'%(uvbin_size/1e3))

        '''
        ax8.errorbar(uvbins, binned_re, yerr=binned_re_err, color='#00FF6C',fmt='.', ecolor='#BDBDBD', alpha=.5,
                     label='%.0f k$\lambda$' % uvbin_size, zorder=10)
        ax8.errorbar(uvbins2, binned_re2, yerr=binned_re_err2, color='r',fmt='.', ecolor='#00AAFF', alpha=.5,
                     label='%.0f k$\lambda$' % uvbin_size2, zorder=10)
        '''

        ax3.axhline(0, c='k', ls='--', zorder=10)
        ax8.axhline(1, c='k', ls='--', label='SNR = 1', zorder=10)
        eff_res = 2.6e6 #1.93e6 #2.45e6 #2.45e6
        ax5.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
        ax2.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
        ax10.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
        ax8.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
        ax5.text(.775, .02, '  Effective\nresolution', transform=ax5.transAxes, fontsize=20, color='#FF8B00')

        ax1.legend(fontsize=20, ncol=1)
        ax3.legend(fontsize=20, ncol=1)
        ax6.legend(fontsize=20, ncol=1)
        ax2.legend(fontsize=20, ncol=1)
        ax8.legend(fontsize=20)
        ax10.legend(fontsize=20)
        ax5.legend(fontsize=20)

        ax3.set_xlim(xlo1, xhi1)
        ax6.set_xlim(xlo1, xhi1)
        ax3.set_ylim(-.52,.52)
        ax5.set_ylim(-.025,.025)
        ax2.set_ylim(1.1e-7,.5)
        xlo, xhi = [4e4, 3e6] #ax5.get_xlim()
        #xlo, xhi = ax2.get_xlim()
        ax5.set_xlim(xlo, xhi)
        ax2.set_xlim(xlo, xhi)
        ax10.set_xlim(xlo, xhi)
        ax8.set_xlim(xlo, xhi)
        ax8.set_ylim(1e-2,5)
        ax6.set_yscale('log')
        ax5.set_xscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax8.set_xscale('log')
        ax8.set_yscale('log')
        ax10.set_xscale('log')
        ax10.set_yscale('log')
        #ax3.set_yticks([-.5,0,.5])
        ax10.set_yticks([1e-5,1e-1,1e3])
        ax6.set_yticks([1e-4,1e-2,1e0])

        ax1.set_ylabel('Brightness\n[$10^{10}$ Jy sr$^{-1}$]', fontsize=24)
        ax6.set_ylabel('Brightness\n[$10^{10}$ Jy sr$^{-1}$]', fontsize=24)
        ax3.set_xlabel('r ["]', fontsize=24)
        ax3.set_ylabel('Normalized\nresidual', fontsize=24)
        ax5.set_ylabel('Re(V) [Jy]', fontsize=24)
        ax2.set_ylabel('Re(V) [Jy]', fontsize=24)
        ax10.set_ylabel('Norm. resid.', fontsize=24)
        ax8.set_ylabel('SNR', fontsize=24)
        ax8.set_xlabel(r'Baseline [$\lambda$]', fontsize=24)
        ax1.text(.9, .2, 'a)', transform=ax1.transAxes, fontsize=24)
        ax6.text(.9, .82, 'b)', transform=ax6.transAxes, fontsize=24)
        ax3.text(.9, .82, 'c)', transform=ax3.transAxes, fontsize=24)
        ax5.text(.1, .05, 'd)', transform=ax5.transAxes, fontsize=24)
        ax2.text(.1, .8, 'e)', transform=ax2.transAxes, fontsize=24)
        ax10.text(.07, .1, 'f)', transform=ax10.transAxes, fontsize=24)
        ax8.text(.07, .82, 'g)', transform=ax8.transAxes, fontsize=24)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax5.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax6.get_xticklabels(), visible=False)
        plt.setp(ax10.get_xticklabels(), visible=False)

        """


        #ax8.text(.1, .1, 'f)', transform=ax8.transAxes, fontsize=24)

        #ax9.set_ylim(1e-3, 1e1)

        #ax10.semilogx(uvbins, resid_vis, '.', c='#33D9FF')


        """
        '''
        ax4 = fig1.add_subplot(gs1_single[5])
        resid = binned_re - GPHF.HankelTransform(uvbins)
        rmse = (np.mean(resid ** 2)) ** .5
        ax4.semilogx(uvbins, (binned_re - GPHF.HankelTransform(uvbins)) / max(binned_re), label='RMSE %.2d'% (rmse / max(binned_re)))
        ax4.set_ylabel('Normalized residual', fontsize=24)
        xlo, xhi = ax4.get_xlim()
        ax4.set_xlim(xlo, xhi)
        #plt.legend(fontsize=10)
        plt.setp(ax4.get_xticklabels(), visible=False)
        '''
        '''
        ax4 = fig1.add_subplot(gs1_double[9])
        plt.ylabel('Power', fontsize=24)

        #gs_cbar = GridSpec(2, 8, right=.99)
        plt.loglog(uv_pts_toplot[0], pi_toplot[0] * 2 * np.pi, c=cmaps[counter_gp](0))#, label=r'%s bins, $\alpha$ =%s, ss=%.0e, %s iterations'%(nbins, alpha, smooth_strength, smooth_iterations))
        for l in range(len(uv_pts_toplot)):
            plt.loglog(uv_pts_toplot[l], pi_toplot[l] * 2 * np.pi, c=cmaps[counter_gp](l / len(uv_pts_toplot)))
        plt.xlim(xlo, xhi)

        plt.xlim(xlo, 1e7)

        #plt.legend(fontsize=10)
        plt.setp(ax4.get_xticklabels(), visible=False)
        '''
        '''
        ax6 = fig1.add_subplot(gs1_small[12])
        plt.ylabel(r'$\chi^2$', fontsize=24)

        #plt.plot(np.array(chi2s_alt) / len(uv.u), '.', label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        plt.plot(np.array(chi2s) / len(uv.u), label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        #plt.legend(fontsize=10)

        plt.setp(ax6.get_xticklabels(), visible=False)


        ax9 = fig1.add_subplot(gs1_small[14])
        plt.xlabel('Fit iteration #', fontsize=24)
        plt.ylabel('Con.#(i$_{cov}$)', fontsize=24)

        plt.semilogy(condition_numbers, label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        plt.yticks(ax9.get_ylim(), ('%.1e'%ax9.get_ylim()[0], '%.1e'%ax9.get_ylim()[1]))
        '''
        #'''
        """
        #ax8 = fig1.add_subplot(gs1_single[7])
        #ax8 = fig1.add_subplot(gs1_single[5])
        #ax8 = fig1.add_subplot(gs1_double[9])
        ax8 = fig1.add_subplot(gs1_single2[3])
        #ax8.text(.95, .8, 'f)', transform=ax8.transAxes, fontsize=24)
        #ax8.text(.1, .1, 'f)', transform=ax8.transAxes, fontsize=24)
        #plt.ylabel(r'SNR = |Re(V)| $\cdot$ $\sqrt{\rm w}$', fontsize=24)
        plt.ylabel(r'SNR', fontsize=24)
        plt.xlabel(r'Baseline [$\lambda$]', fontsize=24)
        xlo, xhi = ax1.get_xlim()
        ax8.set_xlim(xlo, xhi)

        #ax8.set_xlim(xlo, 4e6)
        #ax8.set_xlim(xlo, 1e7)

        #ax8.set_ylim(1e-1, 1e1)
        #ax8.set_ylim(1e-3, 1e1)
        ax8.set_ylim(0, 5)
        print('start snr')
        if counter_gp == 0:
            #'''

            #plt.legend(fontsize=24, loc=[.5, .05])
        #'''
        """
        '''
        plt.axhline(0, c='#A4A4A4', ls='--')
        plt.xlabel('Baseline [$\lambda$]', fontsize=24)
        plt.ylabel('Im(V) [Jy]', fontsize=24)
        if counter_gp == 0:
            plt.errorbar(uvbins, binned_im, yerr=binned_im_err, marker='.', ls='none', c='k', label='obs, %.0f k$\lambda$ bins'%(uvbin_size/1e3))
        ax8.set_xscale('log')
        plt.xlim(xlo, xhi)
        plt.legend(fontsize=10)
        '''

        '''
        corr = cov_renorm.copy()
        for i in range(len(mu)):
            corr[i] /= np.sqrt(cov_renorm[i, i])
            corr[:, i] /= np.sqrt(cov_renorm[i, i])
        '''
        #"""
        #ax10.axvline(eff_res, zorder=20, c='r', ls='--', label='RMSE[:eff. res.] %.3f'%round(rmse_vis,3))
        #ax8.axvline(eff_res, zorder=20, c='b', ls='--')
        #ax5.legend(loc=[.4,.3])
        #ax10.legend(fontsize=24, ncol=1)
        #"""
        fig1.savefig(plot_msfigs_savefile)#, dpi=600)
        fig2.savefig(plot_msfigs_savefile2)#, dpi=600)
        #plt.show()


def ms_f5(plot_msfigs_savefile, plot_msfigs_savefile2, disc, Rmax, baselines, uv, model, GPHF, mu, nbins, alpha, smooth_strength, uvbin_size, uvbin_size2, uv_pts_toplot, pi_toplot, enforce_positivity, smooth_iterations, cut, counter_gp, known_profile, image_extracted_profile, add_noise, nfits, cmaps, snr_plot_binwidth, snr_plot_binwidth2, plot_savefile, savedir):
        #plt.ion()
        import matplotlib as mpl
        mpl.rcParams['xtick.labelsize'] = 24
        mpl.rcParams['ytick.labelsize'] = 24

        #gs1_single = GridSpec(4, 2, bottom=.18, top=.98, left=.05, right=.97, hspace=0)
        gs1_single = GridSpec(2, 2, bottom=.1, top=.98, left=.09, right=.98, hspace=0, wspace=.23)
        gs1_single2 = GridSpec(2, 2, bottom=.085, top=.98, left=.09, right=.98, hspace=0, wspace=.23)
        gs1_single3 = GridSpec(3, 2, bottom=.085, top=.98, left=.09, right=.98, hspace=0, wspace=.23)
        gs1_single_offset = GridSpec(3, 2, bottom=-0.07, top=.83, left=.08, right=.98, hspace=0, wspace=.23)
        gs1_big = GridSpec(1, 2, bottom=.25, top=.98, left=.09, right=.98, hspace=0, wspace=.23)
        gs1_big2 = GridSpec(1, 2, bottom=.085, top=.75, left=.09, right=.98, hspace=0, wspace=.23)
        gs1_big3 = GridSpec(1, 2, bottom=.30875, top=.75625, left=.09, right=.98, hspace=0, wspace=.23)
        gs1_double = GridSpec(4, 2, bottom=.085, top=.98, left=.09, right=.98, hspace=0, wspace=.23)
        gs1_double_top = GridSpec(4, 2, bottom=.185, top=.98, left=.09, right=.98, hspace=0, wspace=.23)
        gs1_double_mid = GridSpec(4, 2, bottom=-.115, top=1.08, left=.09, right=.98, hspace=0, wspace=.23)
        gs1_double_bot = GridSpec(4, 2, bottom=.085, top=.88, left=.09, right=.98, hspace=0, wspace=.23)
        gs1_double2 = GridSpec(5, 2, bottom=.085, top=.98, left=.09, right=.98, hspace=0, wspace=.23)
        gs1 = GridSpec(4, 2, bottom=.15, top=.98, left=.05, right=.97, hspace=0, wspace=.23)
        gs1_2 = GridSpec(4, 2, bottom=.08, top=.89, left=.05, right=.97, hspace=0, wspace=.23)
        gs1_small = GridSpec(8, 2, bottom=.08, top=.72, left=.05, right=.97, hspace=0, wspace=.23)
        gs2 = GridSpec(4, 2)

        fig1 = plt.figure()
        fig2 = plt.figure()

        grid = np.linspace(0, Rmax, 10000)
        ki = 2*np.pi*np.logspace(np.log10(min(baselines.min(), GPHF.uv_pts[0])),
                                 np.log10(max(baselines.max(), GPHF.uv_pts[-1])),
                                 10**4)

        ax1 = fig1.add_subplot(gs1_single3[0])
        ax6 = fig1.add_subplot(gs1_single3[2])

        '''
        resid = model(GPHF.Rc[:cut]) - mu[:cut]
        rmse = (np.mean(resid ** 2)) ** .5

        clean_r, clean_i, _ = np.genfromtxt(pwd + '/../ALMAsim/' + disc + '/' + disc + '_cleaned_image_radial_profile.txt').T
        interp_func = interp1d(clean_r, clean_i, kind='nearest', bounds_error=False)
        regrid_clean_i = interp_func(GPHF.Rc * rad_to_arcsec)[:-1]

        resid_clean = model(GPHF.Rc[:cut]) - regrid_clean_i
        rmse_clean = (np.nanmean(resid_clean ** 2)) ** .5 # NaNs arise
        # when this profile doesn't extend to as large radii as the fit
        '''

        ax1.plot(grid * rad_to_arcsec, model(grid) / 1e10, 'c', label='CLEAN, Andrews+18')#label='Andrews et al. 2018 (for reference)')
        #ax1.plot(GPHF.Rc[:cut] * rad_to_arcsec, regrid_clean_i / 1e10, c='#a4a4a4', label=r'CLEAN, Briggs 0.5')
        ax1.plot(GPHF.Rc[:cut] * rad_to_arcsec, mu[:cut] / 1e10, c='r', label='Fit')#label=r'Fit to Fedele et al. 2018 data') #AD13FF

        ax6.plot(grid * rad_to_arcsec, model(grid) / 1e10, 'c', label='CLEAN')#label='Andrews et al. 2018 (for reference)')
        #ax6.plot(GPHF.Rc[:cut] * rad_to_arcsec, regrid_clean_i / 1e10, c='#a4a4a4', label=r'CLEAN')
        ax6.plot(GPHF.Rc[:cut] * rad_to_arcsec, mu[:cut] / 1e10, c='r', label='Fit')#label=r'Fit to Fedele et al. 2018 data') #AD13FF

        #plt.errorbar(GPHF.Rc * rad_to_arcsec, mu, yerr=np.diag(cov_renorm)**0.5, fmt='+', label='bins', zorder=-1)
        #plt.fill_between(GPHF.Rc * rad_to_arcsec, mu - np.diag(cov_renorm)**0.5, mu + np.diag(cov_renorm)**0.5, where=  mu - np.diag(cov_renorm)**0.5 >=  mu + np.diag(cov_renorm)**0.5, facecolor='r', alpha=.1)

        xlo1, xhi1 = ax1.get_xlim()

        #ax3 = fig1.add_subplot(gs1_double[2])
        #ax3 = fig1.add_subplot(gs1_double[4])
        ax5 = fig1.add_subplot(gs1_single2[1])
        ax2 = fig1.add_subplot(gs1_single2[3])

        ax10 = fig2.add_subplot(gs1_double_bot[5])
        ax8 = fig2.add_subplot(gs1_double_bot[7])

        uvbins, binned_re, binned_im, binned_re_err, binned_im_err = \
            uv.plot(linestyle='+', color='#A4A4A4',
                label=r'Obs, %.0f k$\lambda$ bins' % (uvbin_size),
                uvbin_size=uvbin_size, fontsize=24, return_binned_values=True)

        uvbins2, binned_re2, binned_im2, binned_re_err2, binned_im_err2 = \
            uv.plot(linestyle='x', linestyle2='+', color='#A4A4A4',
                label=r'obs, %.0f k$\lambda$ bins' % (uvbin_size2),
                uvbin_size=uvbin_size2, fontsize=24, return_binned_values=True)

        def find_nearest(array,value):
        	idx = (np.abs(array - value)).argmin()
        	#print 'nearest value %s. array position %s'%(array[idx],idx)
        	return idx

        snr_barrier = 5e5
        resid_vis = binned_re - GPHF.HankelTransform(uvbins * 2 * np.pi)
        norm_resid_vis = (binned_re - GPHF.HankelTransform(uvbins * 2 * np.pi)) / binned_re
        print('find_nearest',find_nearest(uvbins, snr_barrier))
        resid_vis_to_barrier = resid_vis[:find_nearest(uvbins, snr_barrier)]
        rmse_vis = (np.mean(resid_vis_to_barrier ** 2)) ** .5
        norm_resid_vis_to_barrier = norm_resid_vis[:find_nearest(uvbins, snr_barrier)]
        norm_rmse_vis = (np.mean(norm_resid_vis_to_barrier ** 2)) ** .5
        #print('norm_resid_vis',norm_resid_vis)

        sortidx = np.argsort(baselines)
        bls = baselines[sortidx] / 1e3
        snr = abs(uv.re[sortidx]) / (1 / uv.weights[sortidx] ** .5)
        binwidth = snr_plot_binwidth / 1e3

        bins = np.arange(0, max(bls) + binwidth, binwidth)
        mids = bins[:-1] + binwidth / 2
        bins_snr = np.vstack((np.digitize(bls, bins), snr)).T
        mean_snr = [np.mean(bins_snr[bins_snr[:, 0] == i, 1]) for i in
                    range(len(bins))]  # Runtimewarning here; how to ignore?
        std_snr = [np.std(bins_snr[bins_snr[:, 0] == i, 1]) for i in range(len(bins))]
        ax8.errorbar(mids * 1e3, mean_snr[1:], yerr=std_snr[1:], color='#a4a4a4',fmt='.', ecolor='k',
                     label='%.0f k$\lambda$ bins' % binwidth)

        binwidth = snr_plot_binwidth2 / 1e3
        bins = np.arange(0, max(bls) + binwidth, binwidth)
        mids = bins[:-1] + binwidth / 2
        bins_snr = np.vstack((np.digitize(bls, bins), snr)).T
        mean_snr = [np.mean(bins_snr[bins_snr[:, 0] == i, 1]) for i in
                    range(len(bins))]  # Runtimewarning here; how to ignore?
        std_snr = [np.std(bins_snr[bins_snr[:, 0] == i, 1]) for i in range(len(bins))]
        print('NB the bin centering for SNR plot is wrong; should use the results from uvplot')
        ax8.errorbar(mids * 1e3, mean_snr[1:], yerr=std_snr[1:], color='m',fmt='.', ecolor='#00AAFF', alpha=.5,
                     label='%.0f k$\lambda$' % binwidth, zorder=10)

        #ax3.plot(GPHF.Rc[:cut] * rad_to_arcsec, resid_clean / max(model(GPHF.Rc[:cut])), '.', c='#a4a4a4',
        #     label='CLEAN, RMSE %.3f'% round(rmse_clean / max(model(GPHF.Rc[:cut])),3))
        #ax3.plot(GPHF.Rc[:cut] * rad_to_arcsec, resid[:cut] / max(model(GPHF.Rc[:cut])), '.', c='r', label=r'Fit, RMSE %.3f' % round(rmse / max(model(GPHF.Rc[:cut])),3))

        ax5.plot(uvbins, binned_re, '+', c='c', ms=8, label=r'Obs., %.0f k$\lambda$ bins'%(uvbin_size/1e3))
        ax2.plot(baselines, uv.re, marker='.', ls='none', c='k', label='Obs. > 0') #k
        ax2.plot(baselines, -uv.re, marker='.', ls='none', c='#A4A4A4', label='< 0')
        # plt.errorbar(baselines, uv.re, yerr=uv.weights**-0.5,marker='+', ls='none', c='k', label='obs')
        # plt.errorbar(baselines, -uv.re, yerr=uv.weights**-0.5,marker='x', ls='none', c='c')
        ax2.plot(uvbins, binned_re, marker='+', ms=8, ls='none', color='c', label=r'> 0, %.0f k$\lambda$ bins'%(uvbin_size/1e3)) #33D9FF
        ax2.plot(uvbins, -binned_re, marker='+', ms=8, ls='none', color='#65EB2F', label=r'< 0, %.0f k$\lambda$'%(uvbin_size/1e3)) #9CE73C

        #ax2.errorbar(uvbins, binned_re, yerr=binned_re_err, marker='+', color='r', label=r'Obs. > 0, %.0f k$\lambda$ bins'%(uvbin_size * 50 / 1e3))
        #ax2.errorbar(uvbins, -binned_re, yerr=binned_re_err, marker='x', color='m', label=r'Obs. < 0, %.0f k$\lambda$ bins'%(uvbin_size * 50 / 1e3))
        #plt.errorbar(ki / (2 * np.pi), GPHF.HankelTransform(ki), yerr=np.diag(cov_renorm)**0.5, fmt='+', label='bins', zorder=-1)
        ax5.plot(uvbins2, binned_re2, 'x', c='b', ms=8, label=r'%.0f k$\lambda$'%(uvbin_size2 / 1e3))
        ax5.plot(ki / (2 * np.pi), GPHF.HankelTransform(ki), c='r')#c='#AD13FF')#, label=r'%s bins, $\alpha$ =%s, ss=%.0e' % (nbins, alpha, smooth_strength))
        ax2.plot(uvbins2, binned_re2, marker='x', ms=8, ls='none', c='b', label=r'> 0, %.0f k$\lambda$'%(uvbin_size2 / 1e3))
        ax2.plot(uvbins2, -binned_re2, marker='x', ms=8, ls='none', c='m', label=r'< 0, %.0f k$\lambda$'%(uvbin_size2 / 1e3)) #880901
        plot_log_abs(ki/(2*np.pi), GPHF.HankelTransform(ki), ax2, c='r')#c='#AD13FF')#, alpha=.5)#, label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins,alpha,smooth_strength))

        ax10.plot(uvbins, norm_resid_vis, '+', ms=8, c='#F700FF', label='> 0, %.0f k$\lambda$'%(uvbin_size/1e3))
        ax10.plot(uvbins, -norm_resid_vis, 'x',  ms=8, c='#00FFAA', alpha=.5, label='< 0, %.0f k$\lambda$'%(uvbin_size/1e3))

        '''
        ax8.errorbar(uvbins, binned_re, yerr=binned_re_err, color='#00FF6C',fmt='.', ecolor='#BDBDBD', alpha=.5,
                     label='%.0f k$\lambda$' % uvbin_size, zorder=10)
        ax8.errorbar(uvbins2, binned_re2, yerr=binned_re_err2, color='r',fmt='.', ecolor='#00AAFF', alpha=.5,
                     label='%.0f k$\lambda$' % uvbin_size2, zorder=10)
        '''

        #ax3.axhline(0, c='k', ls='--', zorder=10)
        ax8.axhline(1, c='k', ls='--', label='SNR = 1', zorder=10)
        eff_res = 5e6 #1.93e6 #2.45e6 #2.45e6
        ax5.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
        ax2.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
        ax10.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
        ax8.axvline(eff_res, zorder=20, c='#FF8B00', ls='--')
        ax5.text(.7, .05, '  Effective\nresolution', transform=ax5.transAxes, fontsize=20, color='#FF8B00')

        ax1.legend(fontsize=20, ncol=1)
        #ax3.legend(fontsize=20, ncol=1)
        ax6.legend(fontsize=20, ncol=1)
        ax2.legend(fontsize=20, ncol=1)
        #ax8.legend(fontsize=20)
        ax10.legend(fontsize=20)
        ax5.legend(fontsize=20)

        #ax3.set_xlim(xlo1, xhi1)
        ax6.set_xlim(xlo1, xhi1)
        ax6.set_ylim(5e-4, 5)
        #ax3.set_ylim(-.52,.52)
        ax5.set_ylim(-.025,.025)
        ax2.set_ylim(1.1e-7,1)
        xlo, xhi = [9e3, 1e7] #ax5.get_xlim()
        #xlo, xhi = ax2.get_xlim()
        ax5.set_xlim(xlo, xhi)
        ax2.set_xlim(xlo, xhi)
        ax10.set_xlim(xlo, xhi)
        ax8.set_xlim(xlo, xhi)
        ax8.set_ylim(1e-2,5)
        ax6.set_yscale('log')
        ax5.set_xscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax8.set_xscale('log')
        ax8.set_yscale('log')
        ax10.set_xscale('log')
        ax10.set_yscale('log')
        #ax3.set_yticks([-.5,0,.5])
        ax10.set_yticks([1e-5,1e-1,1e3])
        #ax6.set_yticks([1e-4,1e-2,1e0])

        ax1.set_ylabel('Brightness\n[$10^{10}$ Jy sr$^{-1}$]', fontsize=24)
        ax6.set_ylabel('Brightness\n[$10^{10}$ Jy sr$^{-1}$]', fontsize=24)
        ax6.set_xlabel('r ["]', fontsize=24)
        #ax3.set_ylabel('Normalized\nresidual', fontsize=24)
        ax5.set_ylabel('Re(V) [Jy]', fontsize=24)
        ax2.set_ylabel('Re(V) [Jy]', fontsize=24)
        ax10.set_ylabel('Norm. resid.', fontsize=24)
        ax8.set_ylabel('SNR', fontsize=24)
        ax8.set_xlabel(r'Baseline [$\lambda$]', fontsize=24)
        ax1.text(.9, .2, 'a)', transform=ax1.transAxes, fontsize=24)
        ax6.text(.9, .4, 'b)', transform=ax6.transAxes, fontsize=24)
        #ax3.text(.9, .82, 'c)', transform=ax3.transAxes, fontsize=24)
        ax5.text(.8, .9, 'c)', transform=ax5.transAxes, fontsize=24)
        ax2.text(.8, .9, 'd)', transform=ax2.transAxes, fontsize=24)
        ax10.text(.07, .1, 'e)', transform=ax10.transAxes, fontsize=24)
        ax8.text(.5, .1, 'f)', transform=ax8.transAxes, fontsize=24)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax5.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        #plt.setp(ax6.get_xticklabels(), visible=False)
        plt.setp(ax10.get_xticklabels(), visible=False)

        """


        #ax8.text(.1, .1, 'f)', transform=ax8.transAxes, fontsize=24)

        #ax9.set_ylim(1e-3, 1e1)

        #ax10.semilogx(uvbins, resid_vis, '.', c='#33D9FF')


        """
        '''
        ax4 = fig1.add_subplot(gs1_single[5])
        resid = binned_re - GPHF.HankelTransform(uvbins)
        rmse = (np.mean(resid ** 2)) ** .5
        ax4.semilogx(uvbins, (binned_re - GPHF.HankelTransform(uvbins)) / max(binned_re), label='RMSE %.2d'% (rmse / max(binned_re)))
        ax4.set_ylabel('Normalized residual', fontsize=24)
        xlo, xhi = ax4.get_xlim()
        ax4.set_xlim(xlo, xhi)
        #plt.legend(fontsize=10)
        plt.setp(ax4.get_xticklabels(), visible=False)
        '''
        '''
        ax4 = fig1.add_subplot(gs1_double[9])
        plt.ylabel('Power', fontsize=24)

        #gs_cbar = GridSpec(2, 8, right=.99)
        plt.loglog(uv_pts_toplot[0], pi_toplot[0] * 2 * np.pi, c=cmaps[counter_gp](0))#, label=r'%s bins, $\alpha$ =%s, ss=%.0e, %s iterations'%(nbins, alpha, smooth_strength, smooth_iterations))
        for l in range(len(uv_pts_toplot)):
            plt.loglog(uv_pts_toplot[l], pi_toplot[l] * 2 * np.pi, c=cmaps[counter_gp](l / len(uv_pts_toplot)))
        plt.xlim(xlo, xhi)

        plt.xlim(xlo, 1e7)

        #plt.legend(fontsize=10)
        plt.setp(ax4.get_xticklabels(), visible=False)
        '''
        '''
        ax6 = fig1.add_subplot(gs1_small[12])
        plt.ylabel(r'$\chi^2$', fontsize=24)

        #plt.plot(np.array(chi2s_alt) / len(uv.u), '.', label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        plt.plot(np.array(chi2s) / len(uv.u), label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        #plt.legend(fontsize=10)

        plt.setp(ax6.get_xticklabels(), visible=False)


        ax9 = fig1.add_subplot(gs1_small[14])
        plt.xlabel('Fit iteration #', fontsize=24)
        plt.ylabel('Con.#(i$_{cov}$)', fontsize=24)

        plt.semilogy(condition_numbers, label=r'%s bins, $\alpha$ =%s, ss=%.0e'%(nbins, alpha, smooth_strength))
        plt.yticks(ax9.get_ylim(), ('%.1e'%ax9.get_ylim()[0], '%.1e'%ax9.get_ylim()[1]))
        '''
        #'''
        """
        #ax8 = fig1.add_subplot(gs1_single[7])
        #ax8 = fig1.add_subplot(gs1_single[5])
        #ax8 = fig1.add_subplot(gs1_double[9])
        ax8 = fig1.add_subplot(gs1_single2[3])
        #ax8.text(.95, .8, 'f)', transform=ax8.transAxes, fontsize=24)
        #ax8.text(.1, .1, 'f)', transform=ax8.transAxes, fontsize=24)
        #plt.ylabel(r'SNR = |Re(V)| $\cdot$ $\sqrt{\rm w}$', fontsize=24)
        plt.ylabel(r'SNR', fontsize=24)
        plt.xlabel(r'Baseline [$\lambda$]', fontsize=24)
        xlo, xhi = ax1.get_xlim()
        ax8.set_xlim(xlo, xhi)

        #ax8.set_xlim(xlo, 4e6)
        #ax8.set_xlim(xlo, 1e7)

        #ax8.set_ylim(1e-1, 1e1)
        #ax8.set_ylim(1e-3, 1e1)
        ax8.set_ylim(0, 5)
        print('start snr')
        if counter_gp == 0:
            #'''

            #plt.legend(fontsize=24, loc=[.5, .05])
        #'''
        """
        '''
        plt.axhline(0, c='#A4A4A4', ls='--')
        plt.xlabel('Baseline [$\lambda$]', fontsize=24)
        plt.ylabel('Im(V) [Jy]', fontsize=24)
        if counter_gp == 0:
            plt.errorbar(uvbins, binned_im, yerr=binned_im_err, marker='.', ls='none', c='k', label='obs, %.0f k$\lambda$ bins'%(uvbin_size/1e3))
        ax8.set_xscale('log')
        plt.xlim(xlo, xhi)
        plt.legend(fontsize=10)
        '''

        '''
        corr = cov_renorm.copy()
        for i in range(len(mu)):
            corr[i] /= np.sqrt(cov_renorm[i, i])
            corr[:, i] /= np.sqrt(cov_renorm[i, i])
        '''
        #"""
        #ax10.axvline(eff_res, zorder=20, c='r', ls='--', label='RMSE[:eff. res.] %.3f'%round(rmse_vis,3))
        #ax8.axvline(eff_res, zorder=20, c='b', ls='--')
        #ax5.legend(loc=[.4,.3])
        #ax10.legend(fontsize=24, ncol=1)
        #"""
        fig1.savefig(plot_msfigs_savefile)#, dpi=600)
        fig2.savefig(plot_msfigs_savefile2)#, dpi=600)
        #plt.show()
