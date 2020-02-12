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

import matplotlib.pyplot as plt

def plot_brightness_profile(fit_r, fit_i, ax, yscale='linear', comparison_profile=None):
    """ # TODO: add docstring
    """

    ax.plot(fit_r, fit_i / 1e10, 'r', label='Frank')

    if comparison_profile:
        ax.plot(comparison_profile[0], comparison_profile[1] / 1e10, 'c', label='Comparison profile')

    ax.set_xlabel('r ["]')
    ax.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
    ax.set_yscale(yscale)


def plot_binned_vis(baselines, vis, vis_err, ax, xscale='log', yscale='linear', plot_CIs=False):
    """ # TODO: add docstring
    """
    if plot_CIs:
        ax.errorbar(baselines, vis, yerr=vis_err, fmt='r.', ecolor='#A4A4A4', label=r'Obs., %s k$\lambda$ bins'%binwidth)
    else:
        ax.plot(baselines, vis)

    ax.set_xlabel(r'Baseline [$\lambda$]')
    ax.set_ylabel('V [Jy]')
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)


def plot_vis_fit(baselines, vis_fit, ax, xscale='log', yscale='linear',
                            comparison_profile=None):
    """ # TODO: add docstring
    """
    ax.plot(baselines, vis_fit, 'r')

    if comparison_profile:
        ax.plot(comparison_profile[0], comparison_profile[1], 'c', label='DHT of comparison profile')

    ax.set_xlabel(r'Baseline [$\lambda$]')
    ax.set_ylabel('Re(V) [Jy]')
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)


def plot_vis_resid(baselines, vis, vis_err, ax, xscale='log', yscale='linear', normalize_resid=True):
    """ # TODO: add docstring
    """
    resid = obs - fit
    if normalize_resid: resid /= max(obs)
    rmse = (mean(resid**2))**.5

    ax.plot(baselines, resid, label='RMSE %.3f'%rmse)

    ax.set_xlabel(r'Baseline [$\lambda$]')
    if normalize_resid: ax.set_ylabel('Normalized\nresidual')
    else: ax.set_ylabel('Residual [Jy]')
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
