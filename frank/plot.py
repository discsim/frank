# Frankenstein: 1D disc brightness profile reconstruction from Fourier data
# using non-parametric Gaussian Processes
#
# Copyright (C) 2019-2020  R. Booth, J. Jennings, M. Tazzari
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
"""This module contains plotting routines for visualizing and analyzing
Frankenstein fits.
"""
import numpy as np
import matplotlib.pyplot as plt
from frank.useful_funcs import sweep_profile

__all__ = ['plot_brightness_profile', 'plot_vis_fit', 'plot_vis',
          'plot_vis_resid', 'plot_pwr_spec', 'plot_2dsweep']

def plot_brightness_profile(fit_r, fit_i, ax, yscale='linear',c='r', ls='-', ylolim=None, comparison_profile=None):
    """ # TODO: add docstring
    """
    ax.plot(fit_r, fit_i / 1e10, c=c, ls=ls, label='Frank')

    if comparison_profile:
        ax.plot(comparison_profile[0], comparison_profile[1] / 1e10, 'c', label='Comparison profile')

    ax.set_xlabel('r ["]')
    ax.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
    ax.set_yscale(yscale)
    if ylolim: ax.set_ylim(bottom=ylolim)
    ax.legend()

    if yscale == 'linear': ax.axhline(0, c='c', ls='--', zorder=10)

def plot_vis_fit(baselines, vis_fit, ax, c='r', c2='#1EFEDC', ls='-', ls2='None', xscale='log', yscale='linear', ylolim=None,
                            comparison_profile=None):
    """ # TODO: add docstring
    """
    if yscale == 'linear':
        ax.plot(baselines, vis_fit, c=c, ls=ls, label='Frank')
        ax.axhline(0, c='c', ls='--', zorder=10)
    if yscale == 'log':
        ax.plot(baselines, vis_fit, c=c, ls=ls, label='Frank>0')
        ax.plot(baselines, -vis_fit, c=c2, ls=ls2, label='Frank<0')

    if comparison_profile: # TODO: update
        if yscale == 'linear':
            ax.plot(comparison_profile[0], comparison_profile[1], '#8E44AD', label='DHT of comparison profile')
        if yscale == 'log':
            ax.plot(comparison_profile[0], comparison_profile[1], '#8E44AD', label='DHT of comparison profile')

    ax.set_xlabel(r'Baseline [$\lambda$]')
    ax.set_ylabel('Re(V) [mJy]')
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.legend()

    if ylolim: ax.set_ylim(bottom=ylolim)

def plot_vis(baselines, vis, vis_err, ax, c='k', c2=None, marker='.', marker2='.', ls='None', ls2='None', binwidth='unspecified', xscale='log', yscale='linear',
             plot_CIs=False, zoom=None, ylabel='Re(V) [mJy]'):
    """ # TODO: add docstring
    """
    if plot_CIs: # TODO: update for log scale
        ax.errorbar(baselines, vis, yerr=vis_err, color=c, marker=ms, ecolor='#A4A4A4', label=r'Obs., %.0f k$\lambda$ bins'%binwidth/1e3)
    else:
        if yscale == 'linear':
            ax.plot(baselines, vis, c=c, marker=marker, ls=ls, label=r'Obs., %.0f k$\lambda$ bins'%(binwidth/1e3))
            ax.axhline(0, c='c', ls='--', zorder=10)
        if yscale == 'log':
            ax.plot(baselines, vis, c=c, marker=marker, ls=ls, label=r'Obs.>0, %.0f k$\lambda$ bins'%(binwidth/1e3))
            ax.plot(baselines, -vis, c=c2, marker=marker2, ls=ls2, label=r'Obs.<0, %.0f k$\lambda$ bins'%(binwidth/1e3))

    ax.set_xlabel(r'Baseline [$\lambda$]')
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.legend()

    if zoom is not None: ax.set_ylim(zoom)

def plot_vis_resid(baselines, obs, fit, ax, c='k', marker='.', ls='None', binwidth='unspecified', xscale='log', yscale='linear', normalize_resid=False):
    """ # TODO: add docstring
    """
    resid = obs - fit
    if normalize_resid: resid /= obs
    rmse = (np.mean(resid**2))**.5

    ax.plot(baselines, resid, c=c, marker=marker, ls=ls, label=r'%.0f k$\lambda$ bins, RMSE %.3f'%(binwidth/1e3,rmse))

    ax.set_xlabel(r'Baseline [$\lambda$]')
    if normalize_resid: ax.set_ylabel('Normalized\nresidual')
    else: ax.set_ylabel('Residual [mJy]')
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.legend()

    if yscale == 'linear':
        ax.axhline(0, c='c', ls='--', zorder=10)
        ax.set_ylim(-2 * rmse, 2 * rmse)

def plot_pwr_spec(baselines, pwr_spec, ax, c='#B123D7', ls='-', ylolim=None, xscale='log', yscale='log'):
    ax.plot(baselines, pwr_spec, c=c, ls=ls)

    ax.set_xlabel(r'Baseline [$\lambda$]')
    ax.set_ylabel(r'Power [Jy$^2$]') # TODO: update units / label
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    if ylolim: ax.set_ylim(bottom=ylolim)

def plot_2dsweep(r, I, ax, cax=None, cmap='inferno', vmin=None, vmax=None):
    I2D, xmax, ymax = sweep_profile(r, I)
    I2D /= 1e10
    ax.imshow(I2D, origin='lower', extent=[xmax,-xmax,-ymax,ymax], vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlabel('RA offset ["]')
    ax.set_ylabel('Dec offset ["]')

    import matplotlib.colors as mpl_cs
    from matplotlib import cm
    norm = mpl_cs.Normalize(vmin=I2D.min(), vmax=I2D.max())
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array([])

    cbar = plt.colorbar(m, ax=ax, orientation='vertical', shrink=1.)
    cbar.set_label(r'I [10$^{10}$ Jy sr$^{-1}$]')
