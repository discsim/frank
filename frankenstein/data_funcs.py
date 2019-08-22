import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
pwd = os.getcwd()

from uvplot import UVTable
from uvplot import COLUMNS_V0, COLUMNS_V2

from constants import rad_to_arcsec, deg_to_rad
from params import savedir, disc # TODO: clean up. if funcs take disc and savedir as input (they should probably), redundant to import here

def arcsec_baseline(x): # provide x as either a disc radius [arcsec] or baseline [lambda]
    return 1 / (x / 60 / 60 * np.pi / 180) # this is obs_wl / (x / 60 / 60 * np.pi / 180) / obs_wl

def convert_units(Rmax, dra, ddec, pa, inc):
    '''Various needed transformations for internal unit consistency. Internal unit for all angular distances is [rad].'''
    Rmax /= rad_to_arcsec
    dra /= rad_to_arcsec
    ddec /= rad_to_arcsec
    pa *= deg_to_rad
    inc *= deg_to_rad

    return Rmax, dra, ddec, pa, inc


def load_obs(uvtable_filename, known_profile):#, add_noise, noise):
    '''Load interferometer obs stored in a UVTable'''
    print('  Loading UVTable(s)',uvtable_filename)
    if len(uvtable_filename) > 1: # combine multiple UVTables
        u, v, re, im, w = [], [], [], [], []
        for i in range(len(uvtable_filename)):
            #uv = UVTable(filename=pwd+'/uvtables/'+uvtable_filename[i], format='ascii', columns=COLUMNS_V0)
            uv = UVTable(filename=pwd + '/../ALMAsim/' + disc + '/' + uvtable_filename[i], format='ascii', columns=COLUMNS_V0)
            # assumes UVTable has columns ['u', 'v', 're', 'im', 'weights']
            u.extend(uv.u)
            v.extend(uv.v)
            re.extend(uv.re)
            im.extend(uv.im)
            w.extend(uv.weights)
            # TODO: add uv.freqs, uv.spw, others?
        savename = uvtable_filename[-1] + '_combined.txt'
        #np.savetxt(pwd + '/uvtables/' + savename, np.stack([u, v, re, im, w], axis=-1))
        np.savetxt(pwd + '/../ALMAsim/' + disc + '/' + savename, np.stack([u, v, re, im, w], axis=-1))
        print('saved combined uv tables as',savename)
        #uv = UVTable(filename=pwd+'/uvtables/'+savename, format='ascii', columns=COLUMNS_V0)
        uv = UVTable(filename=pwd + '/../ALMAsim/' + disc + '/' + savename, format='ascii', columns=COLUMNS_V0)

    else:
        if known_profile:
            #uv = UVTable(filename=pwd + '/../ALMAsim/' + disc + '/' + uvtable_filename[0], format='ascii', columns=COLUMNS_V0)
            #uv = UVTable(filename=pwd + '/uvtables/' + uvtable_filename[0], format='ascii', columns=COLUMNS_V0)
            #uv = UVTable(filename=pwd + '/../ALMAsim/' + disc + '/' + uvtable_filename[0], format='binary', columns=COLUMNS_V2)
            uv = UVTable(filename=pwd+'/uvtables/'+uvtable_filename[0], format='binary', columns=COLUMNS_V2)
        # TODO: get rid of using known_profile just to save in 2 dfft places here and elsewhere
        else:
            #uv = UVTable(filename=pwd+'/uvtables/'+uvtable_filename[0], format='ascii', columns=COLUMNS_V2) # load UVTable
            uv = UVTable(filename=pwd+'/uvtables/'+uvtable_filename[0], format='binary', columns=COLUMNS_V2)
            #uv = UVTable(filename=pwd + '/../ALMAsim/' + disc + '/' + uvtable_filename[0], format='ascii', columns=COLUMNS_V0)
            #uv = UVTable(filename=pwd + '/uvtables/' + uvtable_filename[0], format='ascii', columns=COLUMNS_V0)
            # TODO: add option to pass in as ascii or binary (.npz)

    baselines = np.hypot(uv.u, uv.v)

    return uv, baselines


def find_wcorr(baselines, uv):
    print('    finding normalization to weights, w_{corr}')
    mu, edges = np.histogram(np.log10(baselines), weights=uv.re, bins=300)
    mu2, edges = np.histogram(np.log10(baselines), weights=uv.re ** 2, bins=300)
    N, edges = np.histogram(np.log10(baselines), bins=300)

    centres = 0.5 * (edges[1:] + edges[:-1])

    mu /= np.maximum(N, 1)
    mu2 /= np.maximum(N, 1)

    sigma = (mu2 - mu ** 2) ** 0.5
    wcorr_estimate = sigma[np.where(sigma > 0)].mean()
    '''
    print('    plotting normalization to weights')
    fig = plt.figure()
    gs = GridSpec(2, 1, hspace=0)
    plt.suptitle(disc)
    ax0 = fig.add_subplot(gs[0])
    plt.plot(centres, mu, '+')
    plt.ylabel('I [Jy]')

    ax1 = fig.add_subplot(gs[1])
    plt.plot(centres, sigma, '+', label='mean %.4f Jy'%wcorr_estimate)
    plt.xlabel(r'log$_{10}$(baseline)')
    plt.ylabel(r'$\sigma$ [Jy]')
    plt.legend()

    plt.savefig(savedir + 'wcorr_estimate.png')
    plt.close()
    '''
    return wcorr_estimate
