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
import os
import numpy as np
from scipy.integrate import trapz
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import logging

from frank.utilities import UVDataBinner

from frank.plot import (
    plot_deprojection_uv,
    plot_deprojection_vis,
    plot_brightness_profile,
    plot_confidence_interval,
    plot_vis, plot_vis_fit, plot_vis_resid, plot_vis_hist,
    plot_profile_iterations,
    plot_2dsweep,
    plot_pwr_spec,
    plot_pwr_spec_iterations,
    plot_convergence_criterion
)

# Suppress some benign warnings
import warnings
warnings.filterwarnings('ignore', '.*compatible with tight_layout.*')
warnings.filterwarnings('ignore', '.*handles with labels found.*')


# Global settings for plots
cs = ['#a4a4a4', 'k', '#896360', 'b']
cs2 = ['#3498DB', 'm', '#F9B817', '#ED6EFF']
hist_cs = ['k', 'r', 'g', 'c', 'm', 'b']
ms = ['x', '+', '.', '1']


def frank_plotting_style():
    """Apply custom alterations to the matplotlib style"""
    import frank
    frank_path = os.path.dirname(frank.__file__)
    style_path = os.path.join(frank_path, 'frank.mplstyle')
    plt.style.use(style_path)


def make_deprojection_fig(u, v, up, vp, vis, visp, force_style=True,
                          save_prefix=None):
    r"""
    Produce a simple figure showing the effect of deprojection on the (u, v)
    coordinates and visibilities

    Parameters
    ----------
    u, v : array, unit = :math:`\lambda`
        Projected (u, v) coordinates
    up, vp : array, unit = :math:`\lambda`
        Deprojected (u, v) coordinates
    vis : array, unit = Jy
        Projected visibilities (complex: real + imag * 1j)
    visp : array, unit = Jy
        Deprojected visibilities (complex: real + imag * 1j)
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

    re_vis = np.real(vis)
    re_visp = np.real(visp)

    if force_style:
        frank_plotting_style()

    gs = GridSpec(2, 1)
    fig = plt.figure(figsize=(8, 6))

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    axes = [ax0, ax1]

    plot_deprojection_uv(u / 1e6, v / 1e6, up / 1e6, vp / 1e6, ax0)

    plot_deprojection_vis(u / 1e6, v / 1e6, up / 1e6, vp / 1e6, re_vis * 1e3,
                          re_visp * 1e3, ax1)

    ax0.set_xlabel('u [M$\lambda$]')
    ax0.set_ylabel('v [M$\lambda$]')
    ax1.set_xlabel('Baseline [M$\lambda$]')
    ax1.set_ylabel('Re(V) [mJy]')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=1e-3)

    ax0.legend(loc=0)
    ax1.legend(loc=0)

    plt.tight_layout()

    if save_prefix:
        plt.savefig(save_prefix + '_frank_deprojection.png', dpi=600)
        plt.close()
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

    logging.info('    Making quick figure')

    if force_style:
        frank_plotting_style()

    gs = GridSpec(2, 2, hspace=0)
    fig = plt.figure(figsize=(8, 6))

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[2])

    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[3])

    ax0.text(.5, .9, 'a)', transform=ax0.transAxes)
    ax1.text(.5, .9, 'b)', transform=ax1.transAxes)

    ax2.text(.5, .9, 'c)', transform=ax2.transAxes)
    ax3.text(.5, .9, 'd)', transform=ax3.transAxes)

    axes = [ax0, ax1, ax2, ax3]

    total_flux = trapz(sol.mean * 2 * np.pi * sol.r, sol.r)
    plot_brightness_profile(sol.r, sol.mean / 1e10, ax0, c='r',
        label='frank, total_flux {:.2e} Jy'.format(total_flux))
    plot_brightness_profile(sol.r, sol.mean / 1e10, ax1, c='r', label='frank')

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
        vis_fit = sol.predict_deprojected(binned_vis.uv).real * 1e3

        resid = vis_re_kl - vis_fit
        norm_resid = resid / vis_re_kl
        rmse = (np.mean(resid**2))**.5

        plot_vis(binned_vis.uv, vis_re_kl, vis_err_re_kl, ax2, c=cs[i],
                 marker=ms[i], ls='None',
                 label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))

        plot_vis_resid(binned_vis.uv, resid, ax3, c=cs[i], marker=ms[i],
                       ls='None',
                       label=r'{:.0f} k$\lambda$ bins, RMSE {:.3f}'.format(bin_widths[i]/1e3, rmse))

    vis_fit_kl = sol.predict_deprojected(grid).real * 1e3
    plot_vis_fit(grid, vis_fit_kl, ax2, c='r', label='frank')

    ax1.set_xlabel('r ["]')
    ax0.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
    ax1.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=1e-3)

    ax3.set_xlabel(r'Baseline [$\lambda$]')
    ax2.set_ylabel('Re(V) [mJy]')
    ax3.set_ylabel('Residual [mJy]')
    ax2.set_xscale('log')
    ax3.set_xscale('log')

    xlims = ax2.get_xlim()
    ax3.set_xlim(xlims)

    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.tight_layout()

    if save_prefix:
        plt.savefig(save_prefix + '_frank_fit_quick.png', dpi=600)
        plt.close()
    else:
        plt.show()

    return fig, axes


def make_full_fig(u, v, vis, weights, sol, bin_widths, hyperparameters, dist=None,
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
    hyperparameters : list, len = 2
        Values for the :math:`\alpha` and :math:`w_{smooth}` hyperparameters.
        Used for the plot legends
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

    logging.info('    Making full figure')

    if force_style:
        frank_plotting_style()

    gs = GridSpec(3, 3, hspace=0)
    gs1 = GridSpec(4, 3, hspace=0, top=.88)
    gs2 = GridSpec(3, 3, hspace=.35, left=.04)
    fig = plt.figure(figsize=(8, 6))

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[3])
    ax2 = fig.add_subplot(gs2[6])

    ax3 = fig.add_subplot(gs[1])
    ax4 = fig.add_subplot(gs[4])
    ax5 = fig.add_subplot(gs[7])

    ax6 = fig.add_subplot(gs[2])
    ax7 = fig.add_subplot(gs1[5])
    ax8 = fig.add_subplot(gs1[8])
    ax9 = fig.add_subplot(gs1[11])

    ax0.text(.9, .6, 'a)', transform=ax0.transAxes)
    ax1.text(.9, .6, 'b)', transform=ax1.transAxes)
    ax2.text(.1, .9, 'c)', c='w', transform=ax2.transAxes)

    ax3.text(.1, .5, 'd)', transform=ax3.transAxes)
    ax4.text(.1, .7, 'e)', transform=ax4.transAxes)
    ax5.text(.1, .7, 'f)', transform=ax5.transAxes)
    ax6.text(.9, .6, 'g)', transform=ax6.transAxes)
    ax7.text(.9, .6, 'h)', transform=ax7.transAxes)
    ax8.text(.9, .6, 'i)', transform=ax8.transAxes)
    ax9.text(.9, .6, 'j)', transform=ax9.transAxes)

    axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

    total_flux = trapz(sol.mean * 2 * np.pi * sol.r, sol.r)
    plot_brightness_profile(sol.r, sol.mean / 1e10, ax0, c='r',
        label='frank, flux {:.2e} Jy'.format(total_flux))
    plot_brightness_profile(sol.r, sol.mean / 1e10, ax1, c='r', label='frank')

    u_deproj, v_deproj, vis_deproj = sol.geometry.apply_correction(u, v, vis)
    baselines = (u_deproj**2 + v_deproj**2)**.5
    grid = np.logspace(np.log10(min(baselines.min(), sol.q[0])),
                       np.log10(max(baselines.max(), sol.q[-1])), 10**4
                       )

    ReV = sol.predict_deprojected(grid).real
    zoom_ylim_guess = abs(ReV[np.int(.5 * len(ReV)):]).max()
    zoom_bounds = [-1.1 * zoom_ylim_guess, 1.1 * zoom_ylim_guess]
    ax4.set_ylim(np.multiply(zoom_bounds, 1e3))

    for i in range(len(bin_widths)):
        binned_vis = UVDataBinner(baselines, vis_deproj, weights, bin_widths[i])
        vis_re_kl = binned_vis.V.real * 1e3
        vis_im_kl = binned_vis.V.imag * 1e3
        vis_err_re_kl = binned_vis.error.real * 1e3
        vis_err_im_kl = binned_vis.error.imag * 1e3
        vis_fit = sol.predict_deprojected(binned_vis.uv).real * 1e3

        resid = vis_re_kl - vis_fit
        norm_resid = resid / vis_re_kl
        rmse = (np.mean(resid**2))**.5

        plot_vis(binned_vis.uv, vis_re_kl, vis_err_re_kl, ax3, c=cs[i],
                 marker=ms[i], ls='None',
                 label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))

        plot_vis(binned_vis.uv, vis_re_kl, vis_err_re_kl, ax4, c=cs[i],
                 marker=ms[i], ls='None',
                 label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))

        plot_vis(binned_vis.uv, vis_re_kl, vis_err_re_kl, ax6, c=cs[i],
                 marker=ms[i], ls='None',
                 label=r'Obs.>0, {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))

        plot_vis(binned_vis.uv, -vis_re_kl, -vis_err_re_kl, ax6, c=cs2[i],
                 marker=ms[i], ls='None',
                 label=r'Obs.<0, {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))

        plot_vis(binned_vis.uv, vis_im_kl, vis_err_im_kl, ax9, c=cs[i],
                 marker=ms[i], ls='None',
                 label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))

        plot_vis_resid(binned_vis.uv, resid, ax5, c=cs[i], marker=ms[i], ls='None',
                       label=r'{:.0f} k$\lambda$ bins, RMSE {:.3f}'.format(bin_widths[i]/1e3, rmse))

        edges = np.concatenate([binned_vis.bin_edges[0].data,
                                binned_vis.bin_edges[1].data[-1:]])
        plot_vis_hist(edges, binned_vis.bin_counts.data, ax8,
                      color=hist_cs[i], label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))

    vis_fit_kl = sol.predict_deprojected(grid).real * 1e3
    plot_vis_fit(grid, vis_fit_kl, ax3, c='r', label='frank')
    plot_vis_fit(grid, vis_fit_kl, ax4, c='r', label='frank')
    plot_vis_fit(grid, vis_fit_kl, ax6, c='r', label='frank>0')
    plot_vis_fit(grid, -vis_fit_kl, ax6, c='#1EFEDC', label='frank<0')

    plot_pwr_spec(sol.q, sol.power_spectrum, ax7, label=r'$\alpha$ {:.2f}'.format(
        hyperparameters[0]) + '\n' + '$w_{smooth}$' + ' {:.1e}'.format(hyperparameters[1]))

    plot_2dsweep(sol.r, sol.mean, ax=ax2, cmap='inferno')

    ax1.set_xlabel('r ["]')
    ax0.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
    ax1.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=1e-3)

    ax2.set_xlabel('RA offset ["]')
    ax2.set_ylabel('Dec offset ["]')

    ax3.set_ylabel('Re(V) [mJy]')
    ax4.set_ylabel('Re(V) [mJy]')
    ax5.set_ylabel('Residual [mJy]')
    ax5.set_xlabel(r'Baseline [$\lambda$]')
    ax3.set_xscale('log')
    ax4.set_xscale('log')
    ax5.set_xscale('log')

    ax6.set_ylabel('Re(V) [mJy]')
    ax7.set_ylabel(r'Power [Jy$^2$]')
    ax8.set_ylabel('Count')
    ax9.set_ylabel('Im(V) [mJy]')
    ax9.set_xlabel(r'Baseline [$\lambda$]')
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax7.set_xscale('log')
    ax7.set_yscale('log')
    ax8.set_xscale('log')
    ax8.set_yscale('log')
    ax9.set_xscale('log')
    ax6.set_ylim(bottom=1e-4)

    xlims = ax3.get_xlim()
    ax4.set_xlim(xlims)
    ax5.set_xlim(xlims)
    ax6.set_xlim(xlims)
    ax7.set_xlim(xlims)
    ax8.set_xlim(xlims)
    ax9.set_xlim(xlims)

    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.setp(ax6.get_xticklabels(), visible=False)
    plt.setp(ax7.get_xticklabels(), visible=False)
    plt.setp(ax8.get_xticklabels(), visible=False)

    plt.tight_layout()

    if save_prefix:
        plt.savefig(save_prefix + '_frank_fit_full.png', dpi=600)
        plt.close()
    else:
        plt.show()

    return fig, axes


def make_diag_fig(r, q, iteration_diagnostics, iter_plot_range=None,
                  force_style=True, save_prefix=None):
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
    iter_plot_range : list
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

    logging.info('    Making diagnostic figure')

    if iter_plot_range is None:
        logging.info("      diag_plot is 'true' in your parameter file but"
                     " iter_plot_range is 'null' --> Defaulting to"
                     " plotting all iterations")

        iter_plot_range = [0, iteration_diagnostics['num_iterations']]

    else:
        if iter_plot_range[0] > iteration_diagnostics['num_iterations']:
            logging.info('      iter_plot_range[0] in your parameter file'
                         ' exceeds the number of fit iterations -->'
                         ' Defaulting to plotting all iterations')

            iter_plot_range = [0, iteration_diagnostics['num_iterations']]

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

    ax0.text(.92, .5, 'a)', transform=ax0.transAxes)
    ax1.text(.92, .1, 'b)', transform=ax1.transAxes)

    ax2.text(.05, .5, 'c)', transform=ax2.transAxes)
    ax3.text(.92, .5, 'd)', transform=ax3.transAxes)

    ax4.text(.92, .9, 'e)', transform=ax4.transAxes)

    axes = [ax0, ax1, ax2, ax3, ax4]

    profile_iter = iteration_diagnostics['mean']
    profile_iter_toplot = [x / 1e10 for x in profile_iter]
    pwr_spec_iter = iteration_diagnostics['power_spectrum']
    num_iter = iteration_diagnostics['num_iterations']

    plot_profile_iterations(r, profile_iter_toplot, iter_plot_range, ax0)

    # Plot the difference in the profile between the last 100 iterations
    iter_plot_range_end = [max(iter_plot_range[1] - 100, 0),
                           iter_plot_range[1] - 1]

    plot_profile_iterations(r, np.diff(profile_iter_toplot, axis=0),
                            iter_plot_range_end, ax1,
                            cmap=plt.cm.cividis)  # pylint: disable=no-member

    plot_pwr_spec_iterations(q, pwr_spec_iter, iter_plot_range, ax2)

    # Plot the difference in the power spectrum between the last 100 iterations
    plot_pwr_spec_iterations(q, np.diff(pwr_spec_iter, axis=0),
                             iter_plot_range_end, ax3,
                             cmap=plt.cm.cividis,  # pylint: disable=no-member
                             bbox_x=.45)

    plot_convergence_criterion(profile_iter_toplot, num_iter, ax4, c='k')

    ax0.set_ylabel(r'I [$10^{10}$ Jy sr$^{-1}$]')
    ax1.set_ylabel(r'$I_i - I_{i-1}$ [$10^{10}$ Jy sr$^{-1}$]')
    ax1.set_xlabel('r ["]')

    ax2.set_ylabel(r'Power [Jy$^2$]')
    ax3.set_ylabel(r'PS$_i$ - PS$_{i-1}$ [Jy$^2$]')
    ax3.set_xlabel(r'Baseline [$\lambda$]')
    ax2.set_xscale('log')
    ax3.set_xscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    ax3.set_ylim(bottom=1e-16)

    ax4.set_xlabel('Fit iteration')
    ax4.set_ylabel('Convergence criterion,\n' +
                   r'max(|$I_i - I_{i-1}$|) / max($I_i$)')
    ax4.set_yscale('log')

    xlims = ax2.get_xlim()
    ax3.set_xlim(xlims)

    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.tight_layout()

    if save_prefix:
        plt.savefig(save_prefix + '_frank_fit_diag.png', dpi=600)
        plt.close()
    else:
        plt.show()

    return fig, axes, iter_plot_range


def make_clean_comparison_fig(u, v, vis, weights, sol, mean_convolved, r_clean,
                   I_clean, bin_widths, gamma=1., dist=None, force_style=True,
                   save_prefix=None
                   ):
    r"""
    Produce a figure comparing a frank fit to a CLEAN fit, in real space by
    convolving the frank fit with the CLEAN beam, and in visibility space by
    taking the discrete Hankel transform of the CLEAN profile

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
    mean_convolved : array, unit = Jy / sr
        frank brightness profile convolved with a CLEAN beam
        (see utilities.convolve_profile).
        The assumed unit is for the x-label
    r_clean : array, unit = arcsec
        Radial coordinates at which a supplied CLEAN profile is specified.
        The assumed unit is for the x-label
    I_clean : array, unit = Jy / sr
        CLEAN brightness profile values at coordinates r_clean.
        The assumed unit is for the y-label
    bin_widths : list, unit = \lambda
        Bin widths in which to bin the observed visibilities
    gamma : float, default = 1
        Index of power law normalization to apply to swept profile images'
        colormaps (see matplotlib.colors.PowerNorm).
        gamma=1 yields a linear colormap
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

    gs = GridSpec(3, 1)
    gs2 = GridSpec(3, 3)
    fig = plt.figure(figsize=(12, 15))

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    ax2 = fig.add_subplot(gs2[6])
    ax3 = fig.add_subplot(gs2[7])
    ax4 = fig.add_subplot(gs2[8])

    axes = [ax0, ax1, ax2, ax3, ax4]

    plot_brightness_profile(r_clean, I_clean / 1e10, ax0, c='b', ls='--',
                            label='CLEAN')
    plot_brightness_profile(sol.r, sol.mean / 1e10, ax0, c='r', label='frank')
    plot_brightness_profile(sol.r, mean_convolved / 1e10, ax0, c='k', ls=':',
                            lw=2.5, label='frank, convolved')

    u_deproj, v_deproj, vis_deproj = sol.geometry.apply_correction(u, v, vis)
    baselines = (u_deproj**2 + v_deproj**2)**.5
    grid = np.logspace(np.log10(min(baselines.min(), sol.q[0])),
                       np.log10(max(baselines.max(), sol.q[-1])), 10**4
                       )

    ReV = sol.predict_deprojected(grid).real

    for i in range(len(bin_widths)):
        binned_vis = UVDataBinner(baselines, vis_deproj, weights, bin_widths[i])
        vis_re_kl = binned_vis.V.real * 1e3
        vis_err_re_kl = binned_vis.error.real * 1e3
        vis_fit = sol.predict_deprojected(binned_vis.uv).real * 1e3

        plot_vis(binned_vis.uv, vis_re_kl, vis_err_re_kl, ax1, c=cs[i],
                 marker=ms[i], ls='None',
                 label=r'Obs.>0, {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))
        plot_vis(binned_vis.uv, -vis_re_kl, -vis_err_re_kl, ax1, c=cs2[i],
                 marker=ms[i], ls='None',
                 label=r'Obs.<0, {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))

    vis_fit_kl = sol.predict_deprojected(grid).real * 1e3

    # Take the discrete Hankel transform of the CLEAN profile, using the same
    # collocation points for the DHT as those in the frank fit
    from frank.hankel import DiscreteHankelTransform
    DHT = DiscreteHankelTransform(sol.Rmax, sol.size)
    clean_DHT_kl = sol.predict_deprojected(grid,
                                           I=np.interp(DHT.r, r_clean, I_clean)).real * 1e3

    plot_vis_fit(grid, vis_fit_kl, ax1, c='r', label='frank>0')
    plot_vis_fit(grid, -vis_fit_kl, ax1, c='r', ls='--', label='frank<0')
    plot_vis_fit(grid, clean_DHT_kl, ax1, c='b', label='DHT of CLEAN>0')
    plot_vis_fit(grid, -clean_DHT_kl, ax1, c='b', ls='--', label='DHT of CLEAN<0')

    from matplotlib.colors import PowerNorm
    vmin = 0
    vmax = max(sol.mean.max(), mean_convolved.max(), I_clean.max())
    norm = PowerNorm(gamma, 0, vmax)
    plot_2dsweep(sol.r, sol.mean, ax=ax2, cmap='inferno', norm=norm, vmin=0,
                vmax=vmax / 1e10, xmax=sol.Rmax, plot_colorbar=True)
    plot_2dsweep(sol.r, mean_convolved, ax=ax3, cmap='inferno', norm=norm,
                vmin=0, vmax=vmax / 1e10, xmax=sol.Rmax, plot_colorbar=True)

    # Interpolate the CLEAN profile onto the frank grid to ensure the CLEAN
    # swept 'image' has the same pixel resolution as the frank swept 'images'
    from scipy.interpolate import interp1d
    interp = interp1d(r_clean, I_clean)
    regrid_I_clean = interp(sol.r)
    plot_2dsweep(sol.r, regrid_I_clean, ax=ax4, cmap='inferno', norm=norm,
                vmin=0, vmax=vmax / 1e10, xmax=sol.Rmax, plot_colorbar=True)

    ax0.legend(loc=0)
    ax1.legend(loc=0)

    ax0.set_xlabel('r ["]')
    ax0.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
    ax1.set_xlabel(r'Baseline [$\lambda$]')
    ax1.set_ylabel(r'Re(V) [mJy]')
    ax2.set_xlabel('RA offset ["]')
    ax3.set_xlabel('RA offset ["]')
    ax4.set_xlabel('RA offset ["]')
    ax2.set_ylabel('Dec offset ["]')

    ax0.set_xlim(right=sol.Rmax)
    ax1.set_xlim(.9 * baselines.min(), 1.2 * baselines.max())
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=1e-3)

    ax2.set_title('Unconvolved frank profile swept')
    ax3.set_title('Convolved frank profile swept')
    ax4.set_title('CLEAN profile swept')

    ax0.text(.5, .9, 'a)', transform=ax0.transAxes)
    ax1.text(.5, .9, 'b)', transform=ax1.transAxes)
    ax2.text(.1, .9, 'c)', c='w', transform=ax2.transAxes)
    ax3.text(.1, .9, 'd)', c='w', transform=ax3.transAxes)
    ax4.text(.1, .9, 'e)', c='w', transform=ax4.transAxes)

    if save_prefix:
        plt.savefig(save_prefix + '_frank_clean_comparison.png', dpi=600)
        plt.close()
    else:
        plt.show()

    return fig, axes


def make_overplot_fig(u, v, vis, weights, sol, bin_widths, dist=None,
                   force_style=True, save_prefix=None
                   ):
    r"""
    Produce a figure overplotting multiple fits

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

    logging.info('    Making quick figure')

    if force_style:
        frank_plotting_style()

    gs = GridSpec(2, 2, hspace=0)
    fig = plt.figure(figsize=(8, 6))

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[2])

    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[3])

    ax0.text(.5, .9, 'a)', transform=ax0.transAxes)
    ax1.text(.5, .9, 'b)', transform=ax1.transAxes)

    ax2.text(.5, .9, 'c)', transform=ax2.transAxes)
    ax3.text(.5, .9, 'd)', transform=ax3.transAxes)

    axes = [ax0, ax1, ax2, ax3]

    total_flux = trapz(sol.mean * 2 * np.pi * sol.r, sol.r)
    plot_brightness_profile(sol.r, sol.mean / 1e10, ax0, c='r',
        label='frank, total flux {:.2e} Jy'.format(total_flux))
    plot_brightness_profile(sol.r, sol.mean / 1e10, ax1, c='r', label='frank')

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
        vis_fit = sol.predict_deprojected(binned_vis.uv).real * 1e3

        resid = vis_re_kl - vis_fit
        norm_resid = resid / vis_re_kl
        rmse = (np.mean(resid**2))**.5

        plot_vis(binned_vis.uv, vis_re_kl, vis_err_re_kl, ax2, c=cs[i],
                 marker=ms[i], ls='None', label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))

        plot_vis_resid(binned_vis.uv, resid, ax3, c=cs[i], marker=ms[i],
                       ls='None', label=r'{:.0f} k$\lambda$ bins, RMSE {:.3f}'.format(bin_widths[i]/1e3, rmse))

    vis_fit_kl = sol.predict_deprojected(grid).real * 1e3
    plot_vis_fit(grid, vis_fit_kl, ax2, c='r', label='frank')

    ax1.set_xlabel('r ["]')
    ax0.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
    ax1.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=1e-3)

    ax3.set_xlabel(r'Baseline [$\lambda$]')
    ax2.set_ylabel('Re(V) [mJy]')
    ax3.set_ylabel('Residual [mJy]')
    ax2.set_xscale('log')
    ax3.set_xscale('log')

    xlims = ax2.get_xlim()
    ax3.set_xlim(xlims)

    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.tight_layout()

    if save_prefix:
        plt.savefig(save_prefix + '_frank_fit_quick.png', dpi=600)
        plt.close()
    else:
        plt.show()

    return fig, axes


def make_bootstrap_fig(r, profiles, dist=None, force_style=True,
                       save_prefix=None):
    r"""
    Produce a figure showing a bootstrap analysis for a Frankenstein fit

    Parameters
    ----------
    r : array, unit = arcsec
        Single set of radial collocation points used in all bootstrap fits
    profiles : array, unit = Jy / sr
        Brightness profiles of all bootstrap fits
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

    logging.info(' Making bootstrap summary figure')

    if force_style:
        frank_plotting_style()

    gs = GridSpec(2, 2, hspace=0)
    fig = plt.figure(figsize=(8, 6))

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[2])

    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[3])

    axes = [ax0, ax1, ax2, ax3]

    ax0.text(.6, .9, 'a) Bootstrap: {} trials'.format(len(profiles)),
             transform=ax0.transAxes)
    ax1.text(.9, .7, 'b)', transform=ax1.transAxes)
    ax2.text(.9, .9, 'c)', transform=ax2.transAxes)
    ax3.text(.9, .7, 'd)', transform=ax3.transAxes)

    mean_profile = np.mean(profiles, axis=0)
    std = np.std(profiles, axis=0)

    plot_confidence_interval(r, (mean_profile - std) / 1e10, (mean_profile + std) /
                             1e10, ax1, color='r', alpha=.7, label='Stan. dev. of trials')
    plot_confidence_interval(r, (mean_profile - std) / 1e10, (mean_profile + std) /
                             1e10, ax3, color='r', alpha=.7, label='Stan. dev. of trials')

    for i in range(len(profiles)):
        plot_brightness_profile(r, profiles[i] / 1e10, ax0, c='k', alpha=.2)
        plot_brightness_profile(r, profiles[i] / 1e10, ax2, c='k', alpha=.2)

    plot_brightness_profile(r, mean_profile / 1e10, ax1,
                            c='#35F9E1', label='Mean of trials')
    plot_brightness_profile(r, mean_profile / 1e10, ax3,
                            c='#35F9E1', label='Mean of trials')

    ax0.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
    ax1.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
    ax2.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
    ax3.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
    ax1.set_xlabel('r ["]')
    ax3.set_xlabel('r ["]')

    ax2.set_yscale('log')
    ax3.set_yscale('log')
    ax2.set_ylim(bottom=1e-4)
    ax3.set_ylim(bottom=1e-4)

    plt.tight_layout()

    if save_prefix:
        plt.savefig(save_prefix + '_frank_bootstrap.png', dpi=600)
        plt.close()
    else:
        plt.show()

    return fig, axes
