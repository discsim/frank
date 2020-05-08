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

# Suppress some benign warnings
import warnings
warnings.filterwarnings('ignore', '.*compatible with tight_layout.*')
warnings.filterwarnings('ignore', '.*handles with labels found.*')

from frank.utilities import sweep_profile


def plot_deprojection_uv(u, v, up, vp, ax):
    """
    Overplot projected and deprojected (u, v) coordinates

    Parameters
    ----------
    u, v : array
        Projected (u, v) coordinates
    up, vp : array
        Deprojected (u, v) coordinates
    ax : Matplotlib `~.axes.Axes` class
        Axis on which to plot
    """

    ax.plot(u, v, '+', c='#1EC8FE', label='Projected')
    ax.plot(up, vp, 'x', c='#D14768', label='Deprojected')

    ax.legend(loc='best')


def plot_deprojection_vis(u, v, up, vp, vis, visp, ax):
    r"""
    Overplot projected and deprojected visibilities

    Parameters
    ----------
    u, v : array
        Projected (u, v) coordinates
    up, vp : array
        Deprojected (u, v) coordinates
    vis : array
        Projected visibilities (complex: real + imag * 1j)
    visp : array
        Deprojected visibilities (complex: real + imag * 1j)
    ax : Matplotlib `~.axes.Axes` class
        Axis on which to plot
    """

    # Projected baselines
    bs = np.hypot(u, v)
    # Deprojected baselines
    bsp = np.hypot(up, vp)

    ax.plot(bs, vis, '+', c='#1EC8FE', label='Projected')
    ax.plot(bsp, vis, 'x', c='#D14768', label='Deprojected')

    ax.legend(loc='best')


def plot_brightness_profile(fit_r, fit_i, ax, dist=None, **kwargs):
    """
    Plot a brightness profile as a function of disc radius, I(r)

    Parameters
    ----------
    fit_r : array
        Radial data coordinates
    fit_i : array
        Brightness values at fit_r
    ax : Matplotlib `~.axes.Axes` class
        Axis on which to plot
    dist : float, optional, default = None, unit = [AU]
        Distance to source. If not None, a new `ax` will be created to
        show an upper x-axis in [AU] for the plot on the current `ax`
    """

    if dist:
        ax_new = ax.twiny()
        ax_new.spines['top'].set_color('#1A9E46')
        ax_new.tick_params(axis='x', which='both', colors='#1A9E46')
        ax_new.plot(fit_r * dist, fit_i, **kwargs)
        ax_new.set_xlabel('r [AU]', color='#1A9E46')

    else:
        ax.plot(fit_r, fit_i, **kwargs)

        ax.axhline(0, c='c', ls='--', zorder=10)

        ax.legend(loc='best')


def plot_confidence_interval(fit_r, low_bound, up_bound, ax, **kwargs):
    """
    Plot the confidence interval for a brightness profile fit

    Parameters
    ----------
    fit_r : array
        Radial data coordinates
    low_bound, up_bound : float
        Lower and upper bound of confidence interval for brightness values at
        fit_r
    ax : Matplotlib `~.axes.Axes` class
        Axis on which to plot
    """

    ax.fill_between(fit_r, low_bound, up_bound, **kwargs)

    ax.legend(loc='best')


def plot_vis_fit(baselines, vis_fit, ax, **kwargs):
    r"""
    Plot a visibility domain fit as a function of baseline, V(q)

    Parameters
    ----------
    baselines : array
        Baseline data coordinates
    vis_fit : array
        Visibility amplitude at baselines
    ax : Matplotlib `~.axes.Axes` class
        Axis on which to plot
    """

    ax.plot(baselines, vis_fit, **kwargs)

    ax.axhline(0, c='c', ls='--', zorder=10)

    ax.legend(loc='best')


def plot_vis(baselines, vis, vis_err, ax, plot_CIs=False, **kwargs):
    r"""
    Plot visibility datapoints as a function of baseline, V(q)

    Parameters
    ----------
    baselines : array
        Baseline data coordinates
    vis : array
        Visibility amplitude at baselines
    vis_err : array
        Uncertainty on the visibility amplitudes at baselines
    ax : Matplotlib `~.axes.Axes` class
        Axis on which to plot
    plot_CIs : bool, default = False
        Whether to show confidence intervals on the visibilities
    """

    if plot_CIs:
        ax.errorbar(baselines, vis, yerr=vis_err, **kwargs)
    else:
        ax.plot(baselines, vis, **kwargs)

    ax.axhline(0, c='c', ls='--', zorder=10)

    ax.legend(loc='best')


def plot_vis_resid(baselines, resid, ax, **kwargs):
    r"""
    Plot residuals between visibility datapoints and a visibility domain fit

    Parameters
    ----------
    baselines : array
        Baseline data coordinates
    resid : array
        (Observed - fitted) visibility amplitudes at baselines
    ax : Matplotlib `~.axes.Axes` class
        Axis on which to plot
    """

    ax.plot(baselines, resid, **kwargs)

    ax.legend(loc='best')

    ax.axhline(0, c='c', ls='--', zorder=10)


def plot_vis_hist(bins, counts, ax, **kwargs):
    r"""
    Plot a histogram of visibilities using a precomputed binning

    Parameters
    ----------
    bins : array
        Bin edges
    bin_counts : array
        Number of visibilities in each bin
    ax : Matplotlib `~.axes.Axes` class
        Axis on which to plot
    """

    ax.hist(0.5*(bins[1:]+bins[:-1]), bins, weights=counts, alpha=.5, **kwargs)

    ax.legend(loc='best')


def plot_pwr_spec(baselines, pwr_spec, ax, **kwargs):
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
    ax : Matplotlib `~.axes.Axes` class
        Axis on which to plot
    """

    ax.plot(baselines, pwr_spec, **kwargs)

    ax.legend(loc='best')


def plot_convergence_criterion(profile_iter, N_iter, ax, **kwargs):
    r"""
    Plot the following convergence criterion for a Frankenstein fit,
        :math:`{\rm max}(|I_i - I_{i-1}|) / {\rm max}(I_i)`,
    where :math:`I_i` is the brightness profile at iteration :math:`i`

    Parameters
    ----------
    profile_iter : list, shape = (N_iter, N_coll)
        Brightness profile reconstruction over N_iter iterations.
        N_coll is the number of collocation points, i.e., the number of grid
        points at which the profile is defined
    N_iter : int
        Total number of iterations in the fit
    ax : Matplotlib `~.axes.Axes` class
        Axis on which to plot

    """

    convergence_criterion = []
    for i in range(N_iter):
        this_conv_cri = np.max(np.abs(profile_iter[i] - profile_iter[i-1])) / \
            np.max(profile_iter[i])
        convergence_criterion.append(this_conv_cri)

    ax.plot(range(0, N_iter), convergence_criterion, **kwargs)


def make_colorbar(ax, vmin, vmax, cmap, label, loc=3, bbox_x=.05, bbox_y=.175):
    """
    Custom format to place a colorbar in an inset

    Parameters
    ----------
    ax : Matplotlib `~.axes.Axes` class
        Axis in which to inset the colorbar
    vmin, vmax : int
        Lower and upper bounds of colorbar scale
    cmap : plt.cm colormap
        Colormap to apply to the colorbar
    label : string
        Label for colorbar
    loc : int, one of [1, 2, 3, 4], default = 3
        Quadrant of colorbar in ax
    bbox_x, bbox_y : float, default = 0.05 and 0.175
        x- and y-value where the colorbar is placed
    """

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins1 = inset_axes(ax, width="50%", height="5%", loc=loc,
                        bbox_to_anchor=(bbox_x, bbox_y, 1, 1),
                        bbox_transform=ax.transAxes)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, cax=axins1, orientation="horizontal")
    cbar.set_label(label)
    axins1.xaxis.set_ticks_position("bottom")


def plot_profile_iterations(r, profile_iter, n_iter, ax,
                            cmap=plt.cm.cool,  # pylint: disable=no-member
                            bbox_x=-.02, bbox_y=-.1, **kwargs):
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
    ax : Matplotlib `~.axes.Axes` class
        Axis on which to plot
    cmap : plt.cm colormap, default=plt.cm.cool
        Colormap to apply to the overplotted profiles
    bbox_x, bbox_y : float, default = -0.02 and -0.1
        x- and y-value where the colorbar is placed
    """
    if n_iter[0] >= n_iter[1] or n_iter[1] > len(profile_iter):
        raise ValueError("Require: n_iter[0] < n_iter[1] and"
                         " n_iter[1] <= len(profile_iter)")

    iter_range = range(n_iter[0], n_iter[1])
    for i in iter_range:
        ax.plot(r, profile_iter[i], c=cmap(i / len(iter_range)), **kwargs)
    ax.plot(r, profile_iter[-1], ':', c='k', label='Last iteration', **kwargs)

    make_colorbar(ax, vmin=n_iter[0], vmax=n_iter[1], cmap=cmap,
                  label='Iteration', loc=1, bbox_x=bbox_x, bbox_y=bbox_y)

    ax.legend(loc='best')


def plot_pwr_spec_iterations(q, pwr_spec_iter, n_iter, ax,
                             cmap=plt.cm.cool,  # pylint: disable=no-member
                             bbox_x=.05, bbox_y=.175, **kwargs):
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
    ax : Matplotlib `~.axes.Axes` class
        Axis on which to plot
    cmap : plt.cm colormap, default=plt.cm.cool
        Colormap to apply to the overplotted power spectra
    bbox_x, bbox_y : float, default = 0.05 and 0.175
        x- and y-value where the colorbar is placed
    """

    iter_range = range(n_iter[0], n_iter[1])
    for i in iter_range:
        ax.plot(q, pwr_spec_iter[i], c=cmap(i / len(iter_range)), **kwargs)
    ax.plot(q, pwr_spec_iter[-1], ':', c='k', label='Last iteration', **kwargs)

    make_colorbar(ax, vmin=n_iter[0], vmax=n_iter[1], cmap=cmap,
                  label='Iteration', loc=3, bbox_x=bbox_x, bbox_y=bbox_y)

    ax.legend(loc='best')


def plot_2dsweep(r, I, ax, cmap='inferno', norm=None, vmin=None,
                 vmax=None, xmax=None, plot_colorbar=True, **kwargs):
    r"""
    Plot a radial profile swept over :math:`2 \pi` to produce an image

    Parameters
    ----------
    r : array
        Radial coordinates at which the 1D brightness profile is defined.
        The assumed unit (for the x- and y-label) is arcsec
    I : array
        Brightness values at r. The assumed unit (for the colorbar) is Jy / sr
    ax : Matplotlib `~.axes.Axes` class
        Axis on which to plot
    cmap : Matplotlib colormap, default = 'inferno'
        Colormap to apply to the 2D image
    norm : Matplotlib `colors.Normalize` class
        Colormap normalization for the image and colorbar
    vmin, vmax : float or None (default)
        Lower and upper brightness values (assumed in Jy / sr) for the 2D
        image and colorbar plot's y-axis. If None, they will be set by
        Matplotlib
    xmax : float or None (default)
        Radius at edge of image. If None, it will be set by max(r)
    plot_colorbar: bool, default = True
        Whether to plot a colorbar beside the image
    """

    if xmax is None:
        I2D, xmax, ymax = sweep_profile(r, I)
    else:
        I2D, _, _ = sweep_profile(r, I)
        ymax = xmax * 1

    I2D /= 1e10

    if vmin is None and vmax is None:
        vmin, vmax = I2D.min(), I2D.max()

    ax.imshow(I2D, origin='lower', extent=(xmax, -xmax, -ymax, ymax), vmin=vmin,
              vmax=vmax, cmap=cmap, norm=norm, **kwargs
              )

    # Set a normalization and colormap for the colorbar
    if plot_colorbar:
        import matplotlib.colors as mpl_cs
        from matplotlib import cm
        if norm is None:
            norm = mpl_cs.Normalize(vmin=vmin, vmax=vmax)
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        m.set_array([])

        cbar = plt.colorbar(m, ax=ax, orientation='vertical', shrink=.7)
        cbar.set_label(r'I [$10^{10}$ Jy sr$^{-1}$]')
