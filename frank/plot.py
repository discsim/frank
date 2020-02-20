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

from frank.utilities import sweep_profile

import numpy as np
import matplotlib.pyplot as plt


def plot_brightness_profile(fit_r, fit_i, ax, yscale='linear', c='r', ls='-',
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
    if ylolim:
        ax.set_ylim(bottom=ylolim)
    ax.legend()

    if yscale == 'linear':
        ax.axhline(0, c='c', ls='--', zorder=10)



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

    if ylolim:
        ax.set_ylim(bottom=ylolim)



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
        ax.errorbar(baselines, vis, yerr=vis_err, color=c, marker=marker,
                    ecolor='#A4A4A4',
                    label=r'Obs., {:.0f} k$\lambda$ bins'.format(binwidth/1e3)
                    )
    else:
        if yscale == 'linear':
            ax.plot(baselines, vis, c=c, marker=marker,
                    label=r'Obs., {:.0f} k$\lambda$ bins'.format(binwidth/1e3)
                    )
            ax.axhline(0, c='c', ls='--', zorder=10)

        if yscale == 'log':
            ax.plot(baselines, vis, c=c, marker=marker,
                    label=r'Obs.>0, {:.0f} k$\lambda$ bins'.format(binwidth/1e3)
                    )
            ax.plot(baselines, -vis, c=c2, marker=marker2,
                    label=r'Obs.<0, {:.0f} k$\lambda$ bins'.format(binwidth/1e3)
                    )

    ax.set_xlabel(r'Baseline [$\lambda$]')
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.legend()

    if zoom is not None:
        ax.set_ylim(zoom)



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
    if normalize_resid:
        resid /= obs
    rmse = (np.mean(resid**2))**.5

    ax.plot(baselines, resid, c=c, marker=marker, ls=ls,
            label=r'{:.0f} k$\lambda$ bins, RMSE {:.3f}'.format(binwidth/1e3,
            rmse)
            )

    ax.set_xlabel(r'Baseline [$\lambda$]')
    if normalize_resid:
        ax.set_ylabel('Normalized\nresidual')
    else:
        ax.set_ylabel('Residual [mJy]')
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
          Reconstructed power spectral mode amplitudes at baselines. The assumed
          unit (for the y-label) is Jy^2
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
    ax.set_ylabel(r'Power [Jy$^2$]')
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    if ylolim:
        ax.set_ylim(bottom=ylolim)


def plot_convergence_criterion(profile_iter, N_iter, ax):
    r"""
    Plot a generic convergence criterion for a Frankenstein fit,
        :math:`{\rm max}(|$I_i - I_{i-1}$|) / {\rm max}($I_i$)`,
    where $I_i$ is the brightness profile at iteration $i$

    Parameters
    ----------
    profile_iter : list, shape = (N_iter, N_coll)
          Brightness profile reconstruction over N_iter iterations.
          N_coll is the number of collocation points, i.e., the number of grid
          points at which the profile is defined
    N_iter : int
          Total number of iterations in the fit
    ax : Matplotlib axis
          Axis on which to plot the convergence criterion
    """

    convergence_criterion = []
    for i in range(N_iter):
        this_conv_cri = np.max(np.abs(profile_iter[i] - profile_iter[i-1])) / \
            np.max(profile_iter[i])
        convergence_criterion.append(this_conv_cri)

    ax.plot(range(0, N_iter), convergence_criterion)

    ax.set_xlabel('Fit iteration')
    ax.set_ylabel('Convergence criterion,\n' +
                  r'max(|$I_i - I_{i-1}$|) / max($I_i$)')

    ax.set_yscale('log')


def make_colorbar(ax, vmin, vmax, cmap, label, loc=3, bbox_x=.05, bbox_y=.175):
    """
    Custom format to place a colorbar in an inset

    Parameters
    ----------
    ax : Matplotlib axis
          Axis in which to inset the colorbar
    vmin, vmax : int
          Lower and upper bounds of colorbar scale
    cmap : plt.cm colormap
          Colormap to apply to the colorbar
    label : string
          Label for colorbar
    loc : int, one of [1, 2, 3, 4], default = 3
          Quadrant position of colorbar in ax
    bbox_x, bbox_y : float, default = 0.05 and 0.175
          x- and y-value where the colorbar is placed
    """

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins1 = inset_axes(ax, width="50%", height="5%", loc=loc,
                        bbox_to_anchor=(bbox_x, bbox_y, 1, 1),
                        bbox_transform=ax.transAxes
                        )
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax)
                               )
    cbar = plt.colorbar(sm, cax=axins1, orientation="horizontal")
    cbar.set_label(label)
    axins1.xaxis.set_ticks_position("bottom")


def plot_profile_iterations(r, profile_iter, n_iter, ax,
                            cmap=plt.cm.cool,  # pylint: disable=no-member
                            ylabel=r'I [10$^{10}$ Jy sr$^{-1}$]'
                            ):
    r"""
    Plot a fit's brightness profile reconstruction over a chosen range of
    the fit's iterations

    Parameters
    ----------
    r : array
          Radial data coordinates at which the brightness profile is defined.
          The assumed unit (for the x-label) is arcsec
    profile_iter : list, shape = (n_iter, N_coll)
          Brightness profile reconstruction at each of n_iter iterations. The
          assumed unit (for the y-label) is Jy / sr
    n_iter : list, of the form [start_iteration, stop_iteration]
          Chosen range of iterations in the fit over which to plot profile_iter
    ax : Matplotlib axis
          Axis on which to plot the profile iterations
    cmap : plt.cm colormap, default=plt.cm.cool
          Colormap to apply to the overplotted profiles
    ylabel : string, default = r'I [10$^{10}$ Jy sr$^{-1}$]'
           y-label of the plot
    """
    if n_iter[0] >= n_iter[1] or n_iter[1] > len(profile_iter):
        raise ValueError("Require: n_iter[0] < n_iter[1] and"
                         " n_iter[1] <= len(profile_iter)")

    iter_range = range(n_iter[0], n_iter[1])
    for i in iter_range:
        ax.plot(r, profile_iter[i] / 1e10, c=cmap(i / len(iter_range)))
    ax.plot(r, profile_iter[-1] / 1e10, ':', c='k', label='Last iteration')

    make_colorbar(ax, vmin=n_iter[0], vmax=n_iter[1], cmap=cmap,
                  label='Iteration', loc=1, bbox_x=-.1, bbox_y=-.2
                  )

    ax.legend(loc='upper right')

    ax.set_xlabel('r ["]')
    ax.set_ylabel(ylabel)


def plot_pwr_spec_iterations(q, pwr_spec_iter, n_iter, ax,
                             cmap=plt.cm.cool,  # pylint: disable=no-member
                             ylabel=r'Power [Jy$^2$]'
                             ):
    r"""
    Plot a fit's power spectrum reconstruction over a chosen range of
    the fit's iterations

    Parameters
    ----------
    q : array
          Baselines at which the power spectrum is defined.
          The assumed unit (for the x-label) is :math:`\lambda`
    pwr_spec_iter : list, shape = (n_iter, N_coll)
          Power spectrum reconstruction at each of n_iter iterations. The
          assumed unit (for the y-label) is Jy^2
    n_iter : list, of the form [start_iteration, stop_iteration]
          Chosen range of iterations in the fit over which to plot pwr_spec_iter
    ax : Matplotlib axis
          Axis on which to plot the power spectrum iterations
    ylabel : string, default = r'Power [Jy$^2$]'
           y-label of the plot
    cmap : plt.cm colormap, default=plt.cm.cool
          Colormap to apply to the overplotted power spectra
    """

    iter_range = range(n_iter[0], n_iter[1])
    for i in iter_range:
        ax.plot(q, pwr_spec_iter[i], c=cmap(i / len(iter_range)))
    ax.plot(q, pwr_spec_iter[-1], ':', c='k', label='Last iteration')

    make_colorbar(ax, vmin=n_iter[0], vmax=n_iter[1], cmap=cmap,
                  label='Iteration', loc=3, bbox_x=.05, bbox_y=.175
                  )

    ax.legend(loc='upper right')

    ax.set_ylim(bottom=1e-16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Baseline [$\lambda$]')
    ax.set_ylabel(ylabel)


def plot_convergence_criterion(profile_iter, N_iter, ax):
    r"""
    Plot a generic convergence criterion for a Frankenstein fit,
        :math:`{\rm max}(|$I_i - I_{i-1}$|) / {\rm max}($I_i$)`,
    where $I_i$ is the brightness profile at iteration $i$

    Parameters
    ----------
    profile_iter : list, shape = (N_iter, N_coll)
          Brightness profile reconstruction over N_iter iterations.
          N_coll is the number of collocation points, i.e., the number of grid
          points at which the profile is defined
    N_iter : int
          Total number of iterations in the fit
    ax : Matplotlib axis
          Axis on which to plot the convergence criterion
    """

    convergence_criterion = []
    for i in range(N_iter):
        this_conv_cri = np.max(np.abs(profile_iter[i] - profile_iter[i-1])) / \
        np.max(profile_iter[i])
        convergence_criterion.append(this_conv_cri)

    ax.plot(range(0, N_iter), convergence_criterion)

    ax.set_xlabel('Fit iteration')
    ax.set_ylabel('Convergence criterion,\n' + \
                 r'max(|$I_i - I_{i-1}$|) / max($I_i$)')

    ax.set_yscale('log')


def make_colorbar(ax, vmin, vmax, cmap, label, loc=3, bbox_x=.05, bbox_y=.175):
    """
    Custom format to place a colorbar in an inset

    Parameters
    ----------
    ax : Matplotlib axis
          Axis in which to inset the colorbar
    vmin, vmax : int
          Lower and upper bounds of colorbar scale
    cmap : plt.cm colormap
          Colormap to apply to the colorbar
    label : string
          Label for colorbar
    loc : int, one of [1, 2, 3, 4], default = 3
          Quadrant position of colorbar in ax
    bbox_x, bbox_y : float, default = 0.05 and 0.175
          x- and y-value where the colorbar is placed
    """

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins1 = inset_axes(ax, width="50%", height="5%", loc=loc,
                   bbox_to_anchor=(bbox_x, bbox_y, 1, 1),
                   bbox_transform=ax.transAxes
                   )
    sm = plt.cm.ScalarMappable(cmap=cmap,
                   norm=plt.Normalize(vmin=vmin, vmax=vmax)
                   )
    cbar = plt.colorbar(sm, cax=axins1, orientation="horizontal")
    cbar.set_label(label)
    axins1.xaxis.set_ticks_position("bottom")


def plot_profile_iterations(r, profile_iter, n_iter, ax, cmap=plt.cm.cool,
                            ylabel=r'I [10$^{10}$ Jy sr$^{-1}$]'
                            ):
    r"""
    Plot a fit's brightness profile reconstruction over a chosen range of
    the fit's iterations

    Parameters
    ----------
    r : array
          Radial data coordinates at which the brightness profile is defined.
          The assumed unit (for the x-label) is arcsec
    profile_iter : list, shape = (n_iter, N_coll)
          Brightness profile reconstruction at each of n_iter iterations. The
          assumed unit (for the y-label) is Jy / sr
    n_iter : list, of the form [start_iteration, stop_iteration]
          Chosen range of iterations in the fit over which to plot profile_iter
    ax : Matplotlib axis
          Axis on which to plot the profile iterations
    cmap : plt.cm colormap, default=plt.cm.cool
          Colormap to apply to the overplotted profiles
    ylabel : string, default = r'I [10$^{10}$ Jy sr$^{-1}$]'
           y-label of the plot
    """

    iter_range = range(n_iter[0], n_iter[1])
    for i in iter_range:
        ax.plot(r, profile_iter[i] / 1e10, c=cmap(i / len(iter_range)))
    ax.plot(r, profile_iter[-1] / 1e10, ':', c='k', label='Last iteration')

    make_colorbar(ax, vmin=n_iter[0], vmax=n_iter[1], cmap=cmap,
                  label='Iteration', loc=1, bbox_x=-.1, bbox_y=-.2
                  )

    ax.legend(loc='upper right')

    ax.set_xlabel('r ["]')
    ax.set_ylabel(ylabel)


def plot_pwr_spec_iterations(q, pwr_spec_iter, n_iter, ax, cmap=plt.cm.cool,
                            ylabel=r'Power [Jy$^2$]'
                            ):
    r"""
    Plot a fit's power spectrum reconstruction over a chosen range of
    the fit's iterations

    Parameters
    ----------
    q : array
          Baselines at which the power spectrum is defined.
          The assumed unit (for the x-label) is :math:`\lambda`
    pwr_spec_iter : list, shape = (n_iter, N_coll)
          Power spectrum reconstruction at each of n_iter iterations. The
          assumed unit (for the y-label) is Jy^-2 # TODO: check
    n_iter : list, of the form [start_iteration, stop_iteration]
          Chosen range of iterations in the fit over which to plot pwr_spec_iter
    ax : Matplotlib axis
          Axis on which to plot the power spectrum iterations
    ylabel : string, default = r'Power [Jy$^2$]'
           y-label of the plot
    cmap : plt.cm colormap, default=plt.cm.cool
          Colormap to apply to the overplotted power spectra
    """

    iter_range = range(n_iter[0], n_iter[1])
    for i in iter_range:
        ax.plot(q, pwr_spec_iter[i], c=cmap(i / len(iter_range)))
    ax.plot(q, pwr_spec_iter[-1], ':', c='k', label='Last iteration')

    make_colorbar(ax, vmin=n_iter[0], vmax=n_iter[1], cmap=cmap,
                  label='Iteration', loc=3, bbox_x=.05, bbox_y=.175
                  )

    ax.legend(loc='upper right')

    ax.set_ylim(bottom=1e-16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Baseline [$\lambda$]')
    ax.set_ylabel(ylabel) # TODO: update units / label


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

    ax.imshow(I2D, origin='lower', extent=[xmax, -xmax, -ymax, ymax], vmin=vmin,
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
