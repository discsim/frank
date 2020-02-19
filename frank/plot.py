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

def plot_brightness_profile(fit_r, fit_i, ax, yscale='linear',c='r', ls='-',
                            ylolim=None
                            ):
    """
    Plot a brightness profile as a function of disc radius, I(r)

    Parameters
    ----------
    fit_r : array
          Radial data coordinates. The assumed unit (for the x-label) is arcsec
    fit_i : array
          Brightness values at fit_r. The assumed unit (for the y-label) is
          Jy / sr
    ax : Matplotlib axis
          Axis on which to plot the brightness profile
    yscale : Matplotlib axis scale, default = 'linear'
          Scale for y-axis
    c : Matplotlib color, default = 'r'
          Color of brightness profile line
    ls : Matplotlib line style, default = '-'
          Style of brightness profile line
    ylolim : float, default = None
          Lower limit of plot's y-axis. If None, it will be set by Matplotlib
    """

    ax.plot(fit_r, fit_i / 1e10, c=c, ls=ls, label='Frank')

    ax.set_xlabel('r ["]')
    ax.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
    ax.set_yscale(yscale)
    if ylolim: ax.set_ylim(bottom=ylolim)
    ax.legend()

    if yscale == 'linear': ax.axhline(0, c='c', ls='--', zorder=10)

def plot_vis_fit(baselines, vis_fit, ax, c='r', c2='#1EFEDC', ls='-',
                 ls2='None', xscale='log', yscale='linear', ylolim=None
                 ):
    r"""
    Plot a visibility domain fit as a function of baseline, V(q)

    Parameters
    ----------
    baselines : array
          Baseline data coordinates. The assumed unit (for the x-label) is
          :math:`\lambda`
    vis_fit : array
          Visibility amplitude at baselines. The assumed unit (for the y-label)
          is mJy
    ax : Matplotlib axis
          Axis on which to plot the visibility fit
    c : Matplotlib color, default = 'r'
          Color of visibility fit line
    c2 : Matplotlib color, default = '#1EFEDC'
          Color of negative regions of visibility fit line for log-y plots
    ls : Matplotlib line style, default = '-'
          Style of visibility fit line
    ls2 : Matplotlib line style, default = '-'
          Style of negative regions of visibility fit line for log-y plots
    xscale : Matplotlib axis scale, default = 'log'
          Scale for x-axis
    yscale : Matplotlib axis scale, default = 'linear'
          Scale for y-axis
    ylolim : float, default = None
          Lower limit of plot's y-axis. If None, it will be set by Matplotlib
    """

    if yscale == 'linear':
        ax.plot(baselines, vis_fit, c=c, ls=ls, label='Frank')
        ax.axhline(0, c='c', ls='--', zorder=10)

    if yscale == 'log':
        ax.plot(baselines, vis_fit, c=c, ls=ls, label='Frank>0')
        ax.plot(baselines, -vis_fit, c=c2, ls=ls2, label='Frank<0')

    ax.set_xlabel(r'Baseline [$\lambda$]')
    ax.set_ylabel('Re(V) [mJy]')
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.legend()

    if ylolim: ax.set_ylim(bottom=ylolim)

def plot_vis(baselines, vis, vis_err, ax, c='k', c2='g', marker='.',
             marker2='.', binwidth='unspecified', xscale='log', yscale='linear',
             plot_CIs=False, zoom=None, ylabel='Re(V) [mJy]'
             ):
    r"""
    Plot visibility datapoints as a function of baseline, V(q)

    Parameters
    ----------
    baselines : array
          Baseline data coordinates. The assumed unit (for the x-label) is
          :math:`\lambda`
    vis : array
          Visibility amplitude at baselines. The assumed unit (for the y-label)
          is mJy
    vis_err : array
          Uncertainty on the visibility amplitudes at baselines. The assumed
          unit (for the y-label) is mJy
    ax : Matplotlib axis
          Axis on which to plot the visibilities
    c : Matplotlib color, default = 'k'
          Color of visibility points
    c2 : Matplotlib color, default = 'g'
          Color of negative values of visibility points for log-y plots
    marker : Matplotlib line style, default = '.'
          Marker style of visibility points
    marker2 : Matplotlib line style, default = '.'
          Marker style of negative values of visibility points for log-y plots
    binwidth : int, float or string, default = 'unspecified'
          Width of bins in which data is binned (for legend labels)
    xscale : Matplotlib axis scale, default = 'log'
          Scale for x-axis
    yscale : Matplotlib axis scale, default = 'linear'
          Scale for y-axis
    plot_CIs : bool, default = False
          Whether to show errorbars on the plotted visibilities
    zoom : list = [lower bound, upper bound] or None (default)
           Lower and upper y-bounds for making a zoomed-in plot
    ylabel : string, default = 'Re(V) [mJy]'
           y-label of the plot
    """

    if plot_CIs:
        ax.errorbar(baselines, vis, yerr=vis_err, color=c, marker=ms,
                    ecolor='#A4A4A4',
                    label=r'Obs., %.0f k$\lambda$ bins'%binwidth/1e3
                    )
    else:
        if yscale == 'linear':
            ax.plot(baselines, vis, c=c, marker=marker,
                    label=r'Obs., %.0f k$\lambda$ bins'%(binwidth/1e3)
                    )
            ax.axhline(0, c='c', ls='--', zorder=10)

        if yscale == 'log':
            ax.plot(baselines, vis, c=c, marker=marker,
                    label=r'Obs.>0, %.0f k$\lambda$ bins'%(binwidth/1e3)
                    )
            ax.plot(baselines, -vis, c=c2, marker=marker2,
                    label=r'Obs.<0, %.0f k$\lambda$ bins'%(binwidth/1e3)
                    )

    ax.set_xlabel(r'Baseline [$\lambda$]')
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.legend()

    if zoom is not None: ax.set_ylim(zoom)

def plot_vis_resid(baselines, obs, fit, ax, c='k', marker='.', ls='None',
                   binwidth='unspecified', xscale='log', yscale='linear',
                   normalize_resid=False
                   ):
    r"""
    Plot residuals between visibility datapoints and a visibility domain fit

    Parameters
    ----------
    baselines : array
          Baseline data coordinates. The assumed unit (for the x-label) is
          :math:`\lambda`
    obs : array
          Observed visibility amplitude at baselines.
          The assumed unit (for the y-label) is mJy
    vis : array
          Fitted visibility amplitude at baselines.
          The assumed unit (for the y-label) is mJy
    ax : Matplotlib axis
          Axis on which to plot the residuals
    c : Matplotlib color, default = 'k'
          Color of residual points
    marker : Matplotlib line style, default = '.'
          Marker style of residual points
    ls : Matplotlib line style, default = 'None'
          Style of residual line
    binwidth : int, float or string, default = 'unspecified'
          Width of bins in which data is binned (for legend labels)
    xscale : Matplotlib axis scale, default = 'log'
          Scale for x-axis
    yscale : Matplotlib axis scale, default = 'linear'
          Scale for y-axis
    normalize_resid : bool, default = False
          Whether to plot the residuals normalized pointwise (i.e., locally)
          by the data amplitude
    """

    resid = obs - fit
    if normalize_resid: resid /= obs
    rmse = (np.mean(resid**2))**.5

    ax.plot(baselines, resid, c=c, marker=marker, ls=ls,
            label=r'%.0f k$\lambda$ bins, RMSE %.3f'%(binwidth/1e3,rmse)
            )

    ax.set_xlabel(r'Baseline [$\lambda$]')
    if normalize_resid: ax.set_ylabel('Normalized\nresidual')
    else: ax.set_ylabel('Residual [mJy]')
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.legend()

    if yscale == 'linear':
        ax.axhline(0, c='c', ls='--', zorder=10)
        ax.set_ylim(-2 * rmse, 2 * rmse)

def plot_pwr_spec(baselines, pwr_spec, ax, c='#B123D7', ls='-', ylolim=None,
                  xscale='log', yscale='log'
                  ):
    r"""
    Plot the reconstructed power spectrum of a Frankenstein fit

    Parameters
    ----------
    baselines : array
          Baseline coordinates. The assumed unit (for the x-label) is
          :math:`\lambda`
    pwr_spec : array
          Reconstructed power spectral mode amplitudes at baselines
    ax : Matplotlib axis
          Axis on which to plot the power spectrum
    c : Matplotlib color, default = '#B123D7'
          Color of power spectrum line
    ls : Matplotlib line style, default = '-'
          Style of power spectrum line
    ylolim : float, default = None
          Lower limit of plot's y-axis. If None, it will be set by Matplotlib
    xscale : Matplotlib axis scale, default = 'log'
          Scale for x-axis
    yscale : Matplotlib axis scale, default = 'log'
          Scale for y-axis
    """

    ax.plot(baselines, pwr_spec, c=c, ls=ls)

    ax.set_xlabel(r'Baseline [$\lambda$]')
    ax.set_ylabel(r'Power [Jy$^2$]') # TODO: update units / label
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    if ylolim: ax.set_ylim(bottom=ylolim)

def plot_2dsweep(r, I, ax, cax=None, cmap='inferno', vmin=None, vmax=None):
    r"""
    Plot a radial profile swept over :math:`2 \pi` to produce an image

    Parameters
    ----------
    r : array
          Radial coordinates at which the 1D brightness profile is defined.
          The assumed unit (for the x- and y-label) is arcsec
    I : array
          Brightness values at r. The assumed unit (for the colorbar) is Jy / sr
    ax : Matplotlib axis
          Axis on which to plot the 2D image
    cmap : Matplotlib colormap, default = 'inferno'
          Colormap to apply to the 2D image
    vmin, vmax : float or None (default)
          Lower and upper brightness values (assumed in Jy / sr) for the 2D
          image and colorbar plot's y-axis. If None, they will be set by
          Matplotlib
    """

    I2D, xmax, ymax = sweep_profile(r, I)
    I2D /= 1e10

    ax.imshow(I2D, origin='lower', extent=[xmax,-xmax,-ymax,ymax], vmin=vmin,
              vmax=vmax, cmap=cmap
              )

    ax.set_xlabel('RA offset ["]')
    ax.set_ylabel('Dec offset ["]')

    # set a normalization and colormap for the colorbar
    import matplotlib.colors as mpl_cs
    from matplotlib import cm
    norm = mpl_cs.Normalize(vmin=I2D.min(), vmax=I2D.max())
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array([])

    cbar = plt.colorbar(m, ax=ax, orientation='vertical', shrink=1.)
    cbar.set_label(r'I [10$^{10}$ Jy sr$^{-1}$]')
