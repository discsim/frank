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
from frank import useful_funcs

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

def plot_vis(baselines, vis, vis_err, ax, c='k', marker='.', binwidth='unspecified', xscale='log', yscale='linear',
             plot_CIs=False, zoom=None):
    """ # TODO: add docstring
    """
    if plot_CIs:
        ax.errorbar(baselines, vis, yerr=vis_err, color=c, marker=ms, ecolor='#A4A4A4', label=r'Obs., %.0f k$\lambda$ bins'%binwidth/1e3)
    else:
        ax.plot(baselines, vis, c=c, marker=marker, label=r'Obs., %.0f k$\lambda$ bins'%(binwidth/1e3))

    ax.axhline(0, c='c', ls='--', zorder=10)
    ax.set_xlabel(r'Baseline [$\lambda$]')
    ax.set_ylabel('V [Jy]')
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.legend()

    if yscale == 'linear': ax.axhline(0, c='c', ls='--', zorder=10)

    if zoom: ax.set_ylim(zoom)

def plot_vis_fit(baselines, vis_fit, ax, c='r', ls='-', xscale='log', yscale='linear',
                            comparison_profile=None):
    """ # TODO: add docstring
    """
    ax.plot(baselines, vis_fit, c=c, ls=ls, label='Frank')

    if comparison_profile:
        ax.plot(comparison_profile[0], comparison_profile[1], '#8E44AD', label='DHT of comparison profile')

    ax.axhline(0, c='c', ls='--', zorder=10)
    ax.set_xlabel(r'Baseline [$\lambda$]')
    ax.set_ylabel('Re(V) [Jy]')
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.legend()

    if yscale == 'linear': ax.axhline(0, c='c', ls='--', zorder=10)

def plot_vis_resid(baselines, obs, fit, ax, c='k', marker='.', binwidth='unspecified', xscale='log', yscale='linear', normalize_resid=False):
    """ # TODO: add docstring
    """
    resid = obs - fit
    if normalize_resid: resid /= max(obs)
    rmse = (np.mean(resid**2))**.5

    ax.plot(baselines, resid, c=c, marker=marker, label=r'%.0f k$\lambda$ bins, RMSE %.3f'%(binwidth/1e3,rmse))

    ax.set_xlabel(r'Baseline [$\lambda$]')
    if normalize_resid: ax.set_ylabel('Normalized\nresidual')
    else: ax.set_ylabel('Residual [Jy]')
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.legend()

    if yscale == 'linear': ax.axhline(0, c='c', ls='--', zorder=10)

    ax.set_ylim(-2 * rmse, 2 * rmse)

def plot_2dsweep(brightness, ax):
    im = useful_funcs.create_image(brightness, nxy=1000, dxy=1e-3, Rmin=1e-6, nR=1e5, dR=1e-4, inc=0)
    useful_funcs.show_image(im, ax, origin='lower')
