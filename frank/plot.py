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


def plot_deprojection_effect(u, v, up, vp, vis, visp, axes):
    """
    Overplot projected and deprojected (u, v) coordinates;
    projected and deprojected visibility amplitudes
    (here 'projection' refers to correcting collectively for the source
    inclination, position angle and phase offset)

    Parameters
    ----------
    u, v : array
        Projected (u, v) coordinates
    up, vp : array
        Deprojected (u, v) coordinates
    vis : array
        Projected visibilities (either the real or imaginary component)
    visp : array
        Deprojected visibilities (either the real or imaginary component)
    axes : list[Axes, Axes]
        Three axes on which to plot
    """

    ax0, ax1 = axes

    ax0.plot(u, v, '+', c='#23E1DB', label='Projected')
    ax0.plot(up, vp, 'x', c='#D14768', label='Deprojected')

    # Projected baselines
    bs = np.hypot(u, v)
    # Deprojected baselines
    bsp = np.hypot(up, vp)

    ax1.plot(bs, vis, '+', c='#23E1DB', label='Projected')
    ax1.plot(bsp, visp, 'x', c='#D14768', label='Deprojected')

    ax0.legend(loc='best')
    ax1.legend(loc='best')


def plot_brightness_profile(fit_r, fit_i, ax, dist=None, low_uncer=None,
                            high_uncer=None, **kwargs):
    """
    Plot a brightness profile (and optionally a confidence inverval) as a
    function of disc radius, I(r)

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
    low_uncer : Negative (i.e., below MAP) uncertainty on fit_i
    high_uncer : Positive (i.e., above MAP) uncertainty on fit_i

    Returns
    -------
    ax_new : Matplotlib `~.axes.Axes` class
        Only if dist is not None, the second x-axis, ax_new will be returned

    """

    if dist:
        ax_new = ax.twiny()
        ax_new.spines['top'].set_color('#1A9E46')
        ax_new.tick_params(axis='x', which='both', colors='#1A9E46')
        ax_new.plot(fit_r * dist, fit_i, **kwargs)
        ax_new.set_xlabel('r [AU]', color='#1A9E46')

        return ax_new

    else:
        if low_uncer is not None:
            if high_uncer is None:
                high_uncer = low_uncer * 1.
            ax.fill_between(fit_r, fit_i - low_uncer, fit_i + high_uncer, **kwargs)

        else:
            ax.plot(fit_r, fit_i, **kwargs)

        ax.axhline(0, c='c', ls='--', zorder=10)

        if 'label' in kwargs:
            ax.legend(loc='best')


def plot_vis_quantity(baselines, vis_quantity, ax, vis_quantity_err=None,
                      **kwargs):
    r"""
    Plot a visibility domain quantity (e.g., observed visibilities, a frank fit,
    residual visibilities, a power spectrum) as a function of baseline

    Parameters
    ----------
    baselines : array
        Baseline data coordinates `b`
    vis_quantity : array
        A generic quantity `Q` to plot as a function of baselines `b`, Q(b)
    ax : Matplotlib `~.axes.Axes` class
        Axis on which to plot
    vis_quantity_err : array, optional, default = None
        Uncertainty on vis_quantity values
    """

    # If input arrays are masked with invalid values ('--'), replace those
    # masked values with NaN
    if np.ma.is_masked(baselines):
        baselines = np.ma.array(baselines).filled(np.nan)
        vis_quantity = np.ma.array(vis_quantity).filled(np.nan)
        if vis_quantity_err is not None:
            vis_quantity_err = np.ma.array(vis_quantity_err).filled(np.nan)

    if vis_quantity_err is not None:
        ax.errorbar(baselines, vis_quantity, yerr=vis_quantity_err, **kwargs)
    else:
        ax.plot(baselines, vis_quantity, **kwargs)

    ax.axhline(0, c='c', ls='--', zorder=10)

    if 'label' in kwargs:
        ax.legend(loc='best')


def plot_vis_hist(binned_vis, ax, rescale=None, **kwargs):
    r"""
    Plot a histogram of visibilities using a precomputed binning

    Parameters
    ----------
    binned_vis : UVDataBinner object
        Pre-binned visibilities (see utilities.UVDataBinner)
    ax : Matplotlib `~.axes.Axes` class
        Axis on which to plot
    rescale : float, default=None
        Constant by which to rescale x-values
    """

    edges = np.concatenate([binned_vis.bin_edges[0].data,
                            binned_vis.bin_edges[1].data[-1:]])

    # alter x-axis units
    if rescale is not None:
        edges /= rescale

    counts = binned_vis.bin_counts.data

    ax.hist(0.5 * (edges[1:] + edges[:-1]), edges, weights=counts, alpha=.5, **kwargs)

    if 'label' in kwargs:
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


def plot_iterations(x, iters, n_iter, ax,
                            cmap=plt.cm.cool,  # pylint: disable=no-member
                            bbox_x=-.02, bbox_y=-.1, **kwargs):
    r"""
    Plot a fit quantity (e.g., the brightness profile or power spectrum) over a
    range of the fit's iterations

    Parameters
    ----------
    x : array
        x-values at which to plot (e.g., radii for a brightness profile or
        baselines for a power spectrum)
    iters : list, shape = (n_iter, N_coll)
        Iterations to plot (e.g., brightness profiles or power spectra at each
        of n_iter iterations)
    n_iter : list, of the form [start_iteration, stop_iteration]
        Range of iterations in the fit over which to plot iters
    ax : Matplotlib `~.axes.Axes` class
        Axis on which to plot
    cmap : plt.cm colormap, default=plt.cm.cool
        Colormap to apply to the overplotted profiles
    bbox_x, bbox_y : float, default = -0.02 and -0.1
        x- and y-value where the colorbar is placed
    """
    if n_iter[0] >= n_iter[1] or n_iter[1] > len(iters):
        raise ValueError("Require: n_iter[0] < n_iter[1] and"
                         " n_iter[1] <= len(iters)")

    iter_range = range(n_iter[0], n_iter[1])
    for i in iter_range:
        ax.plot(x, iters[i], c=cmap(i / len(iter_range)), **kwargs)
    ax.plot(x, iters[-1], ':', c='k', label='Last iteration', **kwargs)

    make_colorbar(ax, vmin=n_iter[0], vmax=n_iter[1], cmap=cmap,
                  label='Iteration', loc=1, bbox_x=bbox_x, bbox_y=bbox_y)

    ax.legend(loc='best')


def plot_2dsweep(r, I, ax, cmap='inferno', norm=None, xmax=None, ymax=None,
                 dr=None, plot_colorbar=True, project=False, phase_shift=False,
                 geom=None, cbar_label=r'I [Jy sr$^{-1}$]', **kwargs):
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
        Colormap normalization for the image and colorbar.
    xmax, ymax : float or None (default)
        Value setting the x- and y-bounds of the image (same units as r). The
        positive and negative bounds are both set to this value (modulo sign).
        If not provided, these will be set to r.max()
    dr : float, optional, default = None
        Pixel size (same units as r). If not provided, it will be set at the
        same spatial scale as r
    plot_colorbar: bool, default = True
        Whether to plot a colorbar beside the image
    project : bool, default = False
        Whether to project the swept profile by the supplied geom
    phase_shift : bool, default = False
        Whether to phase shift the projected profile by the supplied geom.
        If False, the source will be centered in the image
    geom : SourceGeometry object, default=None
        Fitted geometry (see frank.geometry.SourceGeometry). Here we use
        geom.inc [deg], geom.PA [deg], geom.dRA [arcsec], geom.dDec [arcsec] if
        project=True
    cbar_label : string, default = r'I [Jy sr$^{-1}$]'
        Colorbar axis label
    """

    I2D, xmax_computed, ymax_computed = sweep_profile(r, I,
                                                    xmax=xmax, ymax=ymax,
                                                    dr=dr,
                                                    project=project,
                                                    phase_shift=phase_shift,
                                                    geom=geom)

    if xmax is None:
        xmax = xmax_computed
    if ymax is None:
        ymax = ymax_computed

    if norm is None:
        import matplotlib.colors as mpl_cs
        norm = mpl_cs.Normalize(vmin=I2D.min(), vmax=I2D.max())

    ax.imshow(I2D, origin='lower', extent=(xmax, -1.0 * xmax, -1.0 * ymax, ymax),
              cmap=cmap, norm=norm, **kwargs
              )

    # Set a normalization and colormap for the colorbar
    if plot_colorbar:
        from matplotlib import cm
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        m.set_array([])

        cbar = plt.colorbar(m, ax=ax, orientation='vertical', shrink=.7)
        cbar.set_label(cbar_label)
