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
"""This module generates figures for a Frankenstein fit and its diagnostics.
"""
from frank.utilities import UVDataBinner
from frank.plot import (
    plot_brightness_profile,
    plot_vis, plot_vis_fit, plot_vis_resid,
    plot_profile_iterations,
    plot_2dsweep,
    plot_pwr_spec,
    plot_pwr_spec_iterations,
    plot_convergence_criterion
)

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Suppress `plt.tight_layout()` warning
import warnings
warnings.filterwarnings('ignore', '.*compatible with tight_layout.*')

# Global settings for plots
cs = ['#a4a4a4', 'k', '#896360', 'b']
cs2 = ['#3498DB', 'm', '#F9B817', '#ED6EFF']
ms = ['x', '+', '.', '1']


def frank_plotting_style():
    """Apply custom alterations to the matplotlib style"""
    import matplotlib as mpl
    mpl.rcParams['font.size'] = 6
    mpl.rcParams['axes.titlesize'] = 6
    mpl.rcParams['axes.labelsize'] = 6
    mpl.rcParams['xtick.labelsize'] = 6
    mpl.rcParams['ytick.labelsize'] = 6
    mpl.rcParams['legend.fontsize'] = 6
    mpl.rcParams['lines.linewidth'] = .5
    mpl.rcParams['lines.markersize'] = 2.
    mpl.rcParams['lines.markeredgewidth'] = .5
    mpl.rcParams['mathtext.default'] = 'regular'
    mpl.rcParams['axes.formatter.min_exponent'] = 2
    mpl.rcParams['axes.formatter.useoffset'] = False
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['xtick.major.size'] = 7
    mpl.rcParams['xtick.minor.size'] = 3.5
    mpl.rcParams['xtick.direction'] = 'inout'
    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['ytick.major.size'] = 7
    mpl.rcParams['ytick.minor.size'] = 3.5
    mpl.rcParams['ytick.direction'] = 'inout'
    mpl.rcParams['ytick.minor.visible'] = True
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['legend.handletextpad'] = 0.4
    mpl.rcParams['figure.subplot.bottom'] = 0.07
    mpl.rcParams['figure.subplot.top'] = 0.93
    mpl.rcParams['errorbar.capsize'] = 5


def make_full_fig(u, v, vis, weights, sol, bin_widths, dist=None,
                  force_style=True, save_prefix=None):
    r"""
      Produce a figure showing a Frankenstein fit and some useful diagnostics

      Parameters
      ----------
      u, v : array, unit = :math:`\lambda`
          u and v coordinates of observations
      vis : array, unit = Jy
          Observed visibilities (complex: real + imag * 1j)
      weights : array, unit = Jy^-2
          Weights assigned to observed visibilities, of the form
          :math:`1 / \sigma^2`
      sol : _HankelRegressor object
          Reconstructed profile using Maximum a posteriori power spectrum
          (see frank.radial_fitters.FrankFitter)
      bin_widths : list, unit = \lambda
          Bin widths in which to bin the observed visibilities
      dist : float, optional, unit = AU, default = None
          Distance to source, used to show second x-axis for brightness profile
      force_style: bool, default = True
          Whether to use preconfigured matplotlib rcParams in generated figure
      save_prefix : string, default = None
          Prefix for saved figure name. If None, the figure won't be saved

    Returns
    -------
    fig : Matplotlib `.Figure` instance
          The produced figure, including the GridSpec
    axes : Matplotlib `~.axes.Axes` class
          The axes of the produced figure
    """

    if force_style:
        frank_plotting_style()

    gs = GridSpec(3, 3, hspace=0)
    gs2 = GridSpec(3, 3, hspace=.35)
    fig = plt.figure(figsize=(8, 6))

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[3])
    ax2 = fig.add_subplot(gs2[6])

    ax3 = fig.add_subplot(gs[1])
    ax4 = fig.add_subplot(gs[4])
    ax5 = fig.add_subplot(gs[7])

    ax6 = fig.add_subplot(gs[2])
    ax7 = fig.add_subplot(gs[5])
    ax8 = fig.add_subplot(gs[8])

    axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

    plot_brightness_profile(sol.r, sol.mean, ax0)
    plot_brightness_profile(sol.r, sol.mean, ax1, yscale='log', ylolim=1e-3)

    u_deproj, v_deproj, vis_deproj = sol.geometry.apply_correction(u, v, vis)
    baselines = (u_deproj**2 + v_deproj**2)**.5
    grid = np.logspace(np.log10(min(baselines.min(), sol.q[0])),
                       np.log10(max(baselines.max(), sol.q[-1])), 10**4
                       )

    ReV = sol.predict_deprojected(grid).real
    zoom_ylim_guess = abs(ReV[np.int(.5 * len(ReV)):]).max()
    zoom_bounds = [-1.1 * zoom_ylim_guess, 1.1 * zoom_ylim_guess]

    for i in range(len(bin_widths)):
        binned_vis = UVDataBinner(
            baselines, vis_deproj, weights, bin_widths[i])
        vis_re_kl = binned_vis.V.real * 1e3
        vis_im_kl = binned_vis.V.imag * 1e3
        vis_err_re_kl = binned_vis.error.real * 1e3
        vis_err_im_kl = binned_vis.error.imag * 1e3

        plot_vis(binned_vis.uv, vis_re_kl,
                 vis_err_re_kl, ax3, c=cs[i], marker=ms[i],
                 binwidth=bin_widths[i])

        plot_vis(binned_vis.uv, vis_re_kl,
                 vis_err_re_kl, ax4, c=cs[i], marker=ms[i],
                 binwidth=bin_widths[i], zoom=np.multiply(zoom_bounds, 1e3))

        plot_vis(binned_vis.uv, vis_re_kl,
                 vis_err_re_kl, ax6, c=cs[i], c2=cs2[i], marker=ms[i],
                 marker2=ms[i], binwidth=bin_widths[i], yscale='log')

        plot_vis(binned_vis.uv, vis_im_kl,
                 vis_err_im_kl, ax8, c=cs[i], marker=ms[i],
                 binwidth=bin_widths[i], ylabel='Im(V) [mJy]')

        plot_vis_resid(binned_vis.uv, vis_re_kl,
                       sol.predict_deprojected(binned_vis.uv).real * 1e3, ax5,
                       c=cs[i], marker=ms[i], binwidth=bin_widths[i],
                       normalize_resid=False)

    vis_fit_kl = sol.predict_deprojected(grid).real * 1e3
    plot_vis_fit(grid, vis_fit_kl, ax3)
    plot_vis_fit(grid, vis_fit_kl, ax4)
    plot_vis_fit(grid, vis_fit_kl, ax6, yscale='log', ylolim=1e-4, ls2='--')

    plot_pwr_spec(sol.q, sol.power_spectrum, ax7)

    plot_2dsweep(sol.r, sol.mean, ax=ax2, cmap='inferno')

    xlims = ax3.get_xlim()
    ax4.set_xlim(xlims)
    ax5.set_xlim(xlims)
    ax6.set_xlim(xlims)
    ax7.set_xlim(xlims)
    ax8.set_xlim(xlims)

    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.setp(ax6.get_xticklabels(), visible=False)
    plt.setp(ax7.get_xticklabels(), visible=False)

    plt.tight_layout()

    if save_prefix:
        plt.savefig(save_prefix + '_frank_fit_full.png', dpi=600)
    else:
        plt.show()

    return fig, axes


def make_quick_fig(u, v, vis, weights, sol, bin_widths, dist=None,
                   force_style=True, save_prefix=None
                   ):
    r"""
    Produce a simple figure showing just a Frankenstein fit, not any diagnostics

    Parameters
    ----------
    u, v : array, unit = :math:`\lambda`
          u and v coordinates of observations
    vis : array, unit = Jy
          Observed visibilities (complex: real + imag * 1j)
    weights : array, unit = Jy^-2
          Weights assigned to observed visibilities, of the form
          :math:`1 / \sigma^2`
    sol : _HankelRegressor object
          Reconstructed profile using Maximum a posteriori power spectrum
          (see frank.radial_fitters.FrankFitter)
    bin_widths : list, unit = \lambda
          Bin widths in which to bin the observed visibilities
    dist : float, optional, unit = AU, default = None
          Distance to source, used to show second x-axis for brightness profile
    force_style: bool, default = True
          Whether to use preconfigured matplotlib rcParams in generated figure
    save_prefix : string, default = None
        Prefix for saved figure name. If None, the figure won't be saved

    Returns
    -------
    fig : Matplotlib `.Figure` instance
          The produced figure, including the GridSpec
    axes : Matplotlib `~.axes.Axes` class
          The axes of the produced figure
    """

    if force_style:
        frank_plotting_style()

    gs = GridSpec(2, 2, hspace=0)
    fig = plt.figure(figsize=(8, 6))

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[2])

    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[3])

    axes = [ax0, ax1, ax2, ax3]

    plot_brightness_profile(sol.r, sol.mean, ax0)
    plot_brightness_profile(sol.r, sol.mean, ax1, yscale='log', ylolim=1e-3)

    u_deproj, v_deproj, vis_deproj = sol.geometry.apply_correction(u, v, vis)
    baselines = (u_deproj**2 + v_deproj**2)**.5
    grid = np.logspace(np.log10(min(baselines.min(), sol.q[0])),
                       np.log10(max(baselines.max(), sol.q[-1])),
                       10**4)

    for i in range(len(bin_widths)):
        binned_vis = UVDataBinner(
            baselines, vis_deproj, weights, bin_widths[i])
        vis_re_kl = binned_vis.V.real * 1e3
        vis_err_re_kl = binned_vis.error.real * 1e3

        plot_vis(binned_vis.uv, vis_re_kl,
                 vis_err_re_kl, ax2, c=cs[i], marker=ms[i],
                 binwidth=bin_widths[i])

        plot_vis_resid(binned_vis.uv, vis_re_kl,
                       sol.predict_deprojected(binned_vis.uv).real * 1e3, ax3,
                       c=cs[i], marker=ms[i], binwidth=bin_widths[i],
                       normalize_resid=False)

    vis_fit_kl = sol.predict_deprojected(grid).real * 1e3
    plot_vis_fit(grid, vis_fit_kl, ax2)

    xlims = ax2.get_xlim()
    ax3.set_xlim(xlims)

    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.tight_layout()

    if save_prefix:
        plt.savefig(save_prefix + '_frank_fit_quick.png', dpi=600)
    else:
        plt.show()

    return fig, axes


def make_diag_fig(r, q, iteration_diagnostics, iter_plot_range=None,
                  force_style=True, save_prefix=None
                  ):
    r"""
    Produce a diagnostic figure showing fit convergence metrics

    Parameters
    ----------
    r : array
          Radial data coordinates at which the brightness profile is defined.
          The assumed unit (for the x-label) is arcsec
    profile_iter : list, shape = (n_iter, N_coll)
          Brightness profile reconstruction at each of n_iter iterations. The
          assumed unit (for the y-label) is Jy / sr
    q : array
          Baselines at which the power spectrum is defined.
          The assumed unit (for the x-label) is :math:`\lambda`
    iteration_diagnostics : dict
          The iteration_diagnositics from FrankFitter
    N_iter : int
          Total number of iterations in the fit
    iter_range : list
          Range of iterations in the fit over which to
          plot brightness profile and power spectrum reconstructions
    force_style: bool, default = True
          Whether to use preconfigured matplotlib rcParams in generated figure
    save_prefix : string, default = None
        Prefix for saved figure name. If None, the figure won't be saved

    Returns
    -------
    fig : Matplotlib `.Figure` instance
          The produced figure, including the GridSpec
    axes : Matplotlib `~.axes.Axes` class
          The axes of the produced figure
    """

    if force_style:
        frank_plotting_style()

    gs = GridSpec(2, 2, hspace=0, bottom=.35)
    gs2 = GridSpec(3, 2, hspace=0, top=.7)
    fig = plt.figure(figsize=(8, 6))

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[2])

    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[3])

    ax4 = fig.add_subplot(gs2[4])

    axes = [ax0, ax1, ax2, ax3, ax4]

    profile_iter = iteration_diagnostics['mean']
    pwr_spec_iter = iteration_diagnostics['power_spectrum']
    num_iter = iteration_diagnostics['num_iterations']

    if iter_plot_range is None:
        iter_plot_range = [0, num_iter]

    plot_profile_iterations(r, profile_iter, iter_plot_range, ax0)

    # Plot the difference in the profile between the last 100 iterations
    iter_plot_range_end = [iter_plot_range[1] - 100, iter_plot_range[1] - 1]

    # pylint: disable=no-member
    plot_profile_iterations(r, np.diff(profile_iter, axis=0) * 1e5,
                            iter_plot_range_end, ax1, cmap=plt.cm.cividis,
                            ylabel=r'$I_i - I_{i-1}$ [$10^{5}$ Jy sr$^{-1}$]'
                            )

    plot_pwr_spec_iterations(q, pwr_spec_iter, iter_plot_range, ax2)

    # Plot the difference in the power spectrum between the last 100 iterations
    plot_pwr_spec_iterations(q, np.diff(pwr_spec_iter, axis=0),
                             iter_plot_range_end, ax3, cmap=plt.cm.cividis,
                             ylabel=r'$PS_i - PS_{i-1}$ [Jy$^2$]', bbox_x=.45
                             )

    plot_convergence_criterion(profile_iter, num_iter, ax4)

    xlims = ax2.get_xlim()
    ax3.set_xlim(xlims)

    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.tight_layout()

    if save_prefix:
        plt.savefig(save_prefix + '_frank_fit_diag.png', dpi=600)
    else:
        plt.show()

    return fig, axes
