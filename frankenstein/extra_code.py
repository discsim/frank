import numpy as np
import matplotlib.pyplot as plt
import os
pwd = os.getcwd()
from scipy.optimize import minimize
from uvplot import UVTable
from uvplot import COLUMNS_V0

#filename = '/uvtables/uvtable_new_nofl_wcorr_30s.txt'
filename = '/uvtables/as209_sc_cont_nofl_chnavg.txt'
#uv = UVTable(filename=pwd+'/uvtables/'+filename, format='ascii', columns=COLUMNS_V0)

u, v, real, imag, weights = np.genfromtxt(pwd+filename, unpack=True)

wle = 0.00130347953005

rad_2_arcsec = ((180 / np.pi) * 3600.)

# Convert u/v to arcsec
u /= wle * rad_2_arcsec
v /= wle * rad_2_arcsec


def apply_phase_shift(u, v, vis, dRA, dDec):
    dRA *= 2. * np.pi
    dDec *= 2. * np.pi

    phi = u * dRA + v * dDec

    return vis * (np.cos(phi) + 1j * np.sin(phi))


def deproject(u, v, inc, pa):
    cos_t = np.cos(pa)
    sin_t = np.sin(pa)

    up = u * cos_t - v * sin_t
    vp = u * sin_t + v * cos_t
    #   De-project
    up *= np.cos(inc)

    return up, vp


def gauss(params):
    dRA, dDec, inc, pa, norm, scal = params

    vis = apply_phase_shift(u, v, real + 1j * imag, dRA, dDec)

    up, vp = deproject(u, v, inc, pa)

    # Evaluate the gaussian:
    gaus = np.exp(- 0.5 * (up ** 2 + vp ** 2) / scal ** 2)

    # Evaluate at the Chi2
    chi2 = weights * np.abs(norm * gaus - vis) ** 2
    return chi2.sum() / (2 * len(weights))


#res = minimize(gauss, [0., 0., 35. * np.pi/ 180, 86. * np.pi / 180, 1., 1.])
res = minimize(gauss, [0., 0., 0., 0., 1., 1.])

# dRA, dDec are in arcsec
#  inc, pa are  in radians
dRA, dDec, inc, pa, norm, scal = res.x

print('dRA, dDec (arcsec):', dRA, dDec)
print('inclination, position angle (deg)', inc * 180 / np.pi, pa * 180 / np.pi)
print('Gaussian normalization, scale factor', norm, scal)
print('Chi2:', res.fun)

# Do a de-projection and plot
visib = apply_phase_shift(u, v, real + 1j * imag, dRA, dDec)
up, vp = deproject(u, v, inc, pa)


"""
# jj: binning visibilities
plt.ion()
baselines = np.hypot(up, vp)
mu, edges = np.histogram(np.log10(baselines), weights=visib, bins=300)
N, edges = np.histogram(np.log10(baselines), bins=300)
centres = 0.5 * (edges[1:] + edges[:-1])
mu /= np.maximum(N, 1)

plt.figure()
plt.plot(centres, mu, '+')
plt.xlabel(r'log$_{10}$(baseline)')
plt.ylabel('I')


# jj: disc transformations as in my code (to check for agreement with above)
def adjust_disc(uv, dra, ddec, inc, pa):
    '''Apply phase shift by dra, ddec to re,im; rotate u,v by pa; deproject u,v by inc'''
    uv.apply_phase(dra, ddec)
    uv.deproject(inc, pa)
    return uv

from uvplot import UVTable
from uvplot import COLUMNS_V0

uv = UVTable(filename=pwd+filename, format='ascii', columns=COLUMNS_V0) # load UVTable
uv.u /= wle
uv.v /= wle
uv = adjust_disc(uv, dRA / rad_2_arcsec, dDec / rad_2_arcsec, inc, pa)

fig = plt.figure()
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 2)
axes = plt.subplot(gs[0]), plt.subplot(gs[1])
axes2 = plt.subplot(gs[2]), plt.subplot(gs[3])
uv.plot(axes=axes, uvbin_size=30e3)#, fig_filename=pwd+'/comparison_test.png', vis_filename=pwd+'/comparison_test.txt')


res = minimize(gauss, [0., 0., 35. * np.pi/ 180, 86. * np.pi / 180, 1., 1.])
#res = minimize(gauss, [0., 0., 0., 0., 1., 1.])

# dRA, dDec are in arcsec
#  inc, pa are  in radians
dRA, dDec, inc, pa, norm, scal = res.x

print('dRA, dDec (arcsec):', dRA, dDec)
print('inclination, position angle (dec)', inc * 180 / np.pi, pa * 180 / np.pi)
print('Gaussian normalization, scale factor', norm, scal)
print('Chi2:', res.fun)

# Do a de-projection and plot
visib = apply_phase_shift(u, v, real + 1j * imag, dRA, dDec)
u2, v2 = deproject(u, v, inc, pa)
plt.subplot(223)
plt.plot(u2/up, '.')
plt.subplot(224)
plt.plot(v2/vp, '.')
print('max',max(np.abs(u2/up)), 'max v', max(np.abs(v2/vp)))
up, vp = deproject(u, v, inc, pa)
uv.plot(axes=axes, color='r', uvbin_size=30e3)#, fig_filename=pwd+'/comparison_test.png', vis_filename=pwd+'/comparison_test.txt')

plt.savefig('test.png')
plt.show()


# manual uvcut
if disc == 'as209_test':
    maxuv = 1600e3
    uvcut = baselines < maxuv

    uv.u = u[uvcut]
    uv.v = v[uvcut]
    uv.re = re[uvcut]
    uv.im = im[uvcut]
    uv.w = w[uvcut]

    u = u[uvcut]
    v = v[uvcut]
    re = re[uvcut]
    im = im[uvcut]
    w = w[uvcut]
    baselines = baselines[uvcut]


if disc == 'as209_test_set_im_to_0':
    im *= 0.
    uv.im *= 0.

'''
if disc == 'as209' or disc == 'citau':
    plt.ylim(-.2e10,3e10) # as2090 mid-res
    plt.xlim(0,1.7) # as209 mid-res
if disc == 'mock1':
    plt.ylim(-.2e10, 1.1e10) # mock1
    plt.xlim(0,.75) # mock1
#plt.ylim(-.5e10,5.5e10) # mock2
#plt.xlim(0,1.1) # mock2
'''


# compare snr for multiple discs:
plt.figure()
colz=['c','r']
alphas = [1.,.3]
for l in range(len(discs_to_compare)):
    #uv = load_obs(uvtable_filenames[l], cross_val, train_or_test, run_stratification)
    uv = UVTable(filename=pwd+'/uvtables/'+uvtable_filenames[l], format='ascii', columns=COLUMNS_V0) # load UVTable
    if l == 0:
        inc, pa, dra, ddec, wle, wcorr = [0.61286176 * 180 / np.pi, 1.50132798 * 180 / np.pi, 1.339188612881619e-06,
                                          1.65779981e-07 * 180 / np.pi, 0.00130347953005, 1.]
        Rmax = 1.6
    else:
        inc, pa, dra, ddec, wle, wcorr = [47.19418051074036, 14.156835217251384,
                                          -0.33340551644789906 / rad_to_arcsec / deg_to_rad,
                                          0.09462042532030927 / rad_to_arcsec / deg_to_rad, 0.00130347953005, 1.]
        Rmax = 1.5
    Rmax, dra, ddec, pa, inc = convert_units(Rmax, dra, ddec, pa, inc)
    uv.u/=wle # TODO: move these into load_obs after removing load_obs below (don't save file with u,v / wle and reload it below)
    uv.v/=wle
    uv = adjust_disc(uv, dra, ddec, inc, pa, wle, to_plot=False)

    #u /= wle
    #v /= wle

    baselines = np.hypot(u, v)

    print('wcorr',wcorr)
    #wcorr = 0.008259489844276307
    w /= wcorr ** 2  # specific to Hankel formalism errors (cf. treatment of 'w' in galario) # TODO: should this be here? I think no


    sortidx = np.argsort(baselines)
    bls = baselines[sortidx] / 1e3
    print('bls',np.shape(bls))
    snr = abs(re[sortidx])  / ( 1 / w[sortidx]**.5)

    bins = np.arange(0, max(bls) + 30, 30) # 30 k\lambda bins
    mids = bins[:-1] + 15
    print('bins',bins)
    idxs = np.digitize(bls, bins)
    #print('idxs',idxs)
    bins_bl = np.vstack((np.digitize(bls, bins), bls)).T
    bins_snr = np.vstack((np.digitize(bls, bins), snr)).T
    mean_bl = [np.mean(bins_bl[bins_bl[:, 0] == i, 1]) for i in range(len(bins))]
    std_bl = [np.std(bins_bl[bins_bl[:, 0] == i, 1]) for i in range(len(bins))]
    mean_snr = [np.mean(bins_snr[bins_snr[:, 0] == i, 1]) for i in range(len(bins))]
    std_snr = [np.std(bins_snr[bins_snr[:, 0] == i, 1]) for i in range(len(bins))]
    #print('bins_bl',bins_bl)
    #print('mean_bl',mean_bl)

    #plt.subplot(211)
    print('disc',discs_to_compare[l])
    plt.errorbar(mids, mean_snr[1:], yerr=std_snr[1:], fmt='none', ecolor=colz[l], alpha=alphas[l], label=discs_to_compare[l])
    if l == 0: plt.axhline(1,c='g',ls='--', label='SNR = 1')
    #plt.xlabel('Baseline [k$\lambda$]')
    plt.xlabel('Baseline [k$\lambda$]')
    plt.ylabel(r'SNR = |Re(V)| $\cdot$ $\sqrt{\rm weights}$')
    plt.legend()

    plt.savefig(savedir+'snr.png')


# Make a fit to the observations
print('fitting Hankel transform to obs...')
tS = time.time()
BMF = BinModel(Rmax, Nbins)
BMF.fit_hankel_coeffs(baselines, uv.re, w)
print ('time to fit', (time.time() - tS), 's\n')

if real_space_fit:
    print('making, plotting BinModel...')
    # Set up a bin model
    BM = BinModel(Rmax, Nbins)
    # Set the data points (bin amplitudes) to be equal to the integral of the profile (at each bin center)
    vals = []
    for r0, r1 in zip(BM.Re[:-1], BM.Re[1:]):
        vals.append(quad(model, r0, r1)[0] / (r1 - r0))
    BM.set_bin_coefficients(vals)  # this sets BM.Inu

    plot_BM(r, BM, baselines, uv.re, ks, Rmax)

print('plotting obs, %s fits...'%ndraws_to_plot)
plot_BMF(r, BMF, baselines, ks, uv.re, w, ndraws_to_plot, Rmax, Nbins, nbins, real_space_fit)

print('finding max likelihood by direct method...')
#find_ML_direct(baselines, uv.re, w, Rmax, r, BM, baselines, uv.re, w, Rmax, r, BMF, prior, real_space_fit)
find_ML_direct(BM) # temporary. should BM be BMF? probably BM?
print('...done')
#plot_ML_direct(Rmax, baselines, uv.re, w, r, real_space_fit)

maxl, samples, chain, burnin = find_ML(BMF, nsteps, Nbins) #BMF or BM?
#np.savez_compressed()
#plot_mcmc(maxl, samples, chain, burnin)


# for plot_snr:
# bins_bl = np.vstack((np.digitize(bls, bins), bls)).T
# mean_bl = [np.mean(bins_bl[bins_bl[:, 0] == i, 1]) for i in range(len(bins))]
# std_bl = [np.std(bins_bl[bins_bl[:, 0] == i, 1]) for i in range(len(bins))]


to_load = ['uvtable_model_temp.txt']  # TODO: stop saving then loading
uv_model = load_obs(to_load, cross_val=False, train_or_test=None, run_stratification=False)


ax = plt.figure().gca()

# BMF_temp = BinModel(Rmax, 5000)
# BMF_temp.set_bin_coefficients(means)


min_corr = []
median_corr = []
std_corr = []
max_abs_cov_norm_by_amp = []
median_cov_norm_by_amp = []

for i in range(len(models_to_compare)):
    # min_corr_i, median_corr_i, std_corr_i, max_abs_cov_norm_by_amp_i, median_cov_norm_by_amp_i = np.loadtxt(savedir + '%s_bin_shift_cov_diag_%sbins_%ssteps.npy' % (disc, models_to_compare[i], niterations))
    # TODO: loading from the above file is defunct
    min_corr.append(min_corr_i)
    median_corr.append(median_corr_i)
    std_corr.append(std_corr_i)
    max_abs_cov_norm_by_amp.append(max_abs_cov_norm_by_amp_i)
    median_cov_norm_by_amp.append(median_cov_norm_by_amp_i)

ax0 = plt.subplot(gs[0, 0])
plt.ylabel('~min(correlation)')
plt.plot(models_to_compare, min_corr, '.', c=cols[0])
plt.plot(models_to_compare, min_corr, '-', c=cols[0])

ax2 = plt.subplot(gs[0, 1])
plt.ylabel('~[max(abs(normed cov))]$^{1/2}$')
plt.axhline(y=10., color='r', linestyle='--', label='test threshold')
plt.semilogy(models_to_compare, np.array(max_abs_cov_norm_by_amp)**.5, '.', c=cols[0])
plt.semilogy(models_to_compare, np.array(max_abs_cov_norm_by_amp)**.5, '-', c=cols[0])
plt.legend()



min_corr = []
median_corr = []
std_corr = []
max_abs_cov_norm_by_amp = []
median_cov_norm_by_amp = []

# degeneracies b/t adjacent bin pairs
fig = plt.figure()  # compare mean fits across models of dfft n_{bins}
plt.suptitle('%s\nscan 0 --> -1: blue --> purple --> pink'%disc)
gs = GridSpec(2, 1, hspace=0)
gs2 = GridSpec(4, 1, hspace=0)
gs_cbar = GridSpec(2, 8, right=.99)

ax1 = fig.add_subplot(gs[0,0])
#plt.xlabel('bin pair [idx]')
plt.ylabel('correlation')
cmap = plt.cm.cool

for i,x in enumerate(corr_adj_bins):
    ax1.plot(np.arange(.5, nbins[ii] - .5), x, c=cmap(i / len(corr_adj_bins)))

    min_corr.append(min(x))
    median_corr.append(np.median(x))
    std_corr.append(np.std(x))
min_corr = min(min_corr)
median_corr = np.mean(median_corr)
std_corr = np.mean(std_corr)

ax2 = fig.add_subplot(gs2[2,0])
plt.ylabel(r'cov / (I$_n$ I$_{n+1}$)')

for i,x in enumerate(cov_norm_by_amps):
    plt.semilogy(np.arange(.5, nbins[ii] - .5), x, 'x', c=cmap(i / len(cov_norm_by_amps)))

    max_abs_cov_norm_by_amp.append(max(abs(x)))
    median_cov_norm_by_amp.append(np.median(abs(x)))
max_abs_cov_norm_by_amp = max(max_abs_cov_norm_by_amp)
median_cov_norm_by_amp = np.mean(median_cov_norm_by_amp)

ax2.set_xscale('linear')

ax2 = fig.add_subplot(gs2[3, 0])
plt.xlabel('bin pair [bin #]')
plt.ylabel(r'-1 $\cdot$ cov / (I$_n$ I$_{n+1}$)')
for i,x in enumerate(cov_norm_by_amps):
    plt.semilogy(np.arange(.5, nbins[ii] - .5), -x, '.', c=cmap(i / len(cov_norm_by_amps)))

plt.savefig(savedir + '%s_bin_correlations_%sbins_%ssteps.png'%(disc, nbins[ii], niterations))

np.savetxt(savedir + '%s_bin_shift_cov_diag_%sbins_%ssteps.npy' % (disc, nbins[ii], niterations), [min_corr, median_corr, std_corr, max_abs_cov_norm_by_amp, median_cov_norm_by_amp])


if disc == 'citau':
    rs, bs = np.genfromtxt('citau_frompaper_digitized.csv', delimiter=',').T
    print('grid',grid)
    integral = 2 * np.pi * trapz(grid * means, grid,
                                 np.diff(grid)[0])  # 2pi correct?
    print('integrated flux', integral)
    interp_func_i = interp1d(grid, means, kind='linear')

    integral_parametric = 2 * np.pi * trapz(rs / 140. * bs, rs / 140., np.diff(rs / 140.)[0])
    print('parametric integral', integral_parametric)
    bs = bs / integral_parametric * integral
    plt.loglog(rs / 140., bs, 'r--')
    plt.xlim(1e-2,1.6)

if disc == 'as209':
    rs, bs = np.genfromtxt('dsharp_as209.csv', delimiter=',').T
    print('grid',grid)
    integral = 2 * np.pi * trapz(grid * means, grid,
                                 np.diff(grid)[0])  # 2pi correct?
    print('integrated flux', integral)
    interp_func_i = interp1d(grid, means, kind='linear')

    integral_parametric = 2 * np.pi * trapz(rs / 121. * bs, rs / 121., np.diff(rs / 121.)[0])
    print('parametric integral', integral_parametric)
    bs = bs / integral_parametric * integral
    plt.plot(rs / 121., bs, 'r--')
    plt.xlim(1e-2,1.6)

if disc == 'as209' or disc == 'citau':
    if disc == 'as209':
        dist = 121. # [pc]
        label_pt1 = 'parametric'
    if disc == 'citau':
        dist = 140. # [pc]
        label_pt1 = 'DSHARP'

'''
if disc == 'as209':
    plt.twinx()
    r_dav, i_dav = np.load(pwd + '/uvtables/davide_profile.npy')
    plt.plot(r_dav, i_dav / integral, 'g', label='Fedele+17')
    #plt.plot(r_dav, i_dav / np.trapz(r_dav * i_dav, r_dav, np.diff(r_dav)[0])**2 * integral)
    print('c',trapz(r_dav * i_dav / max(i_dav) * integral, r_dav, np.diff(r_dav)[0]))
    r_dsharp, i_dsharp = np.genfromtxt(pwd + '/uvtables/dsharp_as209.csv', delimiter=',').T
    r_dsharp_arcsec = r_dsharp/121.
    print('a',integral)
    print('b',trapz(r_dsharp_arcsec * i_dsharp / max(i_dsharp) * integral, r_dsharp_arcsec, np.diff(r_dsharp_arcsec)[0]))
    plt.plot(r_dsharp_arcsec, i_dsharp / integral, 'r', label='DSHARP18')
    #plt.plot(r_dsharp_arcsec, i_dsharp / np.trapz(r_dsharp_arcsec * i_dsharp, r_dsharp_arcsec, np.diff(r_dsharp_arcsec)[0])**2 * integral)
    plt.ylabel('norm. I * integral of 20 bin fit')
    plt.legend(loc='center right', fontsize=10)
    #plt.ylim(-.1, 1)
    #plt.ylim(-.42e-13,3.5e-13) # align y=0 with our bin profiles
    #plt.axhline(0., c='r')
'''


plt.twinx()
plt.plot(nbins, np.array(norm_chi2s) / min(norm_chi2s))
plt.ylabel('$\chi^2 / $ min($\chi^2$)')

plt.twinx()
plt.plot(nbins, bics - min(bics))
plt.ylabel('BIC - min(BIC)')

corr_thresh2 = models_to_compare[next(i for i, x in enumerate(std_corr) if x >= std_corr[0]) - 1]
plt.errorbar(5, corr_thresh2, yerr=1, fmt='o',
             label=r'1 before first time $\sigma_{\rm corr}$ for any bin pair >= that for bins[0,1]), err $\pm 1$')

cov_thresh2 = models_to_compare[next(i for i, x in enumerate(np.array(max_abs_cov_norm_by_amp) ** .5) if x >= 10.) - 1]
plt.errorbar(6, cov_thresh2, yerr=1, fmt='o', label=r'1 before first time max((sqrt(cov / amps)) >= 0.1, err $\pm 1$')


if cross_val:  # TODO: check more carefully that the binning in cross_val is binning by baseline (in bins of constant baseline interval). the bins shouldn't have an equal number of datapoints in them.)

    def stratify_sample(data):
        bins = [data[sort_idxs][i:i + int(ndata / nbins)] for i in range(0, ndata, int(
            ndata / nbins))]  # divide data into nbins of size ndata / nbins, according to baseline

        '''
        if len(bins[-1]) < .5 * len(bins[-2]): # prevent the last bin from only having a few points
            bins[-2] = np.concatenate([bins[-2], bins[-1]])
            bins = bins[:-1]
        '''

        # print('data binned by baselines into chunks of size %s'%np.shape(bins[0]))

        # plt.figure()
        cols = 10 * ['#A4A4A4', '#33D9FF', '#E73CC5', '#FF9700', '#09E02B', 'r', '#E0D905']

        all_train, all_test = [], []
        for i in range(len(bins)):
            train_vals, test_vals = train_test_split(bins[i], test_size=0.2, random_state=46)
            all_train.append(train_vals)
            all_test.append(test_vals)

            # print('all_train',np.shape(all_train[-1]))

            # plt.plot(np.hypot(u_bins[i], v_bins[i]), '.', label=i)
            # plt.legend()
            # plt.savefig(savedir + '%s_data_binned_by_baseline.png' % disc)

        all_train = [x for sl in all_train for x in sl]
        all_test = [x for sl in all_test for x in sl]
        '''
        for k in range(len(idxs_above_thresh)):
            #print('data[idxs_above_thresh][k]', data[idxs_above_thresh][k])
            if data[idxs_above_thresh][k] not in all_train:
                all_train.append(data[idxs_above_thresh][k])
            if data[idxs_above_thresh][k] not in all_test:
                all_test.append(data[idxs_above_thresh][k])
        '''
        return all_train, all_test

    if run_stratification:
        print('len(uv.u)', len(uv.u))

        baselines = np.hypot(uv.u, uv.v)
        sort_idxs = np.argsort(baselines)

        nbins = 1
        ndata = len(uv.u)
        baseline_thresh = 2000
        idxs_above_thresh = [i for i, x in enumerate(baselines) if x >= baseline_thresh]

        plt.figure()
        plt.plot(np.sort(baselines), '.', label='full dataset')
        plt.ylabel('sorted baselines [m]')
        plt.xlabel('idx')

        u_train, u_test = stratify_sample(uv.u)
        v_train, v_test = stratify_sample(uv.v)
        re_train, re_test = stratify_sample(uv.re)
        im_train, im_test = stratify_sample(uv.im)
        w_train, w_test = stratify_sample(uv.weights)

        plt.plot(np.sort((np.array(u_train) ** 2 + np.array(v_train) ** 2) ** .5), 'r+', label='training set')
        plt.plot(np.sort((np.array(u_test) ** 2 + np.array(v_test) ** 2) ** .5), 'gx', label='test set')
        plt.legend()
        # plt.show()
        plt.savefig(savedir + '%s_sorted_baselines.png' % disc)

        np.savetxt(pwd + '/uvtables/' + os.path.splitext(uvtable_filename[-1])[0] + '_train.txt',
                   np.stack([u_train, v_train, re_train, im_train, w_train], axis=-1))
        np.savetxt(pwd + '/uvtables/' + os.path.splitext(uvtable_filename[-1])[0] + '_test.txt',
                   np.stack([u_test, v_test, re_test, im_test, w_test], axis=-1))

    uv = UVTable(filename=pwd + '/uvtables/' + os.path.splitext(uvtable_filename[-1])[0] + '_%s.txt' % train_or_test,
                 format='ascii', columns=COLUMNS_V0)

    chi2_trains.append(chi2)

    uv_model.plot(axes=axes_cv, axes2=axes2, linestyle='-', linestyle2='--', color=cols[i],
                  label='train model, %s bins, $\chi^2$=%.4f' % (models_to_compare[i], chi2 / (2 * len(uv_model.u))),
                  uvbin_size=uvbin_size,
                  fontsize=16, yerr2=False)

    uv.plot(axes=axes_cv, axes2=axes2, linestyle='.', color='k', label='train obs, bin size %s k$\lambda$' % (uvbin_size / 1e3),
            uvbin_size=uvbin_size, fontsize=16)  # , yerr=False)

    interp_cv = interp1d(baselines, model_visibilities, kind='linear')#, bounds_error=False, fill_value=(model_visibilities[0], model_visibilities[-1]))

    # plot the fit of the training set and the test data
    uv = UVTable(filename=pwd + '/uvtables/' + os.path.splitext(uvtable_filename[-1])[0] + '_test.txt',
                 format='ascii', columns=COLUMNS_V0)
    len_test_data = len(uv.u)

    chi2 = galario.double.reduce_chi2(uv.re.copy(order='C'), uv.im.copy(order='C'), uv.weights,
                                      model_visibilities)
    chi2_tests.append(chi2)

    uv.plot(axes=axes3, axes2=axes4, linestyle='.', color='k', label='test obs, bin size %s k$\lambda$' % (uvbin_size / 1e3),
            uvbin_size=uvbin_size, fontsize=16)  # , yerr=False)

    plt.subplot(221)
    plt.xlabel(r'n$_{\rm bins}$')
    plt.ylabel('reduced $\chi^2$')
    plt.plot(models_to_compare, np.array(chi2_trains) / len_train_data / 2, 'k.-', label='train set')
    plt.legend()

    plt.subplot(222)
    plt.xlabel(r'n$_{\rm bins}$')
    plt.ylabel('reduced $\chi^2$')
    plt.plot(models_to_compare, np.array(chi2_tests) / len_test_data / 2, 'r.-', label='test set')
    plt.legend()

    plt.savefig(savedir + '%s_cross_val_nbins%s.png' % (discs_to_compare, models_to_compare))
"""