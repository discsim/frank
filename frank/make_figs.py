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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import PowerNorm
import logging

from frank.utilities import UVDataBinner

from frank.plot import (
    plot_deprojection_effect,
    plot_brightness_profile,
    plot_vis_quantity,
    plot_vis_hist,
    plot_iterations,
    plot_2dsweep,
    plot_convergence_criterion
)

# Suppress some benign warnings
import warnings
warnings.filterwarnings('ignore', '.*handles with labels found.*')


# Global settings for plots
cs = ['#a4a4a4', 'k', '#f781bf', '#dede00']
cs2 = ['#3498DB', '#984ea3', '#4daf4a', '#ff7f00']
hist_cs = ['#e41a1c', '#999999', '#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
           '#984ea3', '#dede00']
multifit_cs = ['#e41a1c', '#999999', '#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
               '#984ea3', '#dede00']
ms = ['x', '+', '.', '1']


class _DoNothingContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource
    def __enter__(self):
        return self.dummy_resource
    def __exit__(self, *args):
        pass

def frank_plotting_style_context_manager(use_frank_style=True):
    """Get a context manager for temporary use of frank's own plotting style"""
    if use_frank_style:
        frank_path = os.path.dirname(__file__)
        style_path = os.path.join(frank_path, 'frank.mplstyle')
        return plt.style.context(style_path)
    else:
        return _DoNothingContextManager()

def use_frank_plotting_style():
    """Set matplotlib to use frank's own plotting style"""
    frank_path = os.path.dirname(__file__)
    style_path = os.path.join(frank_path, 'frank.mplstyle')
    plt.style.use(style_path)


def make_deprojection_fig(u, v, vis, weights, geom, bin_widths, logx=False,
                          logy=False, force_style=True, save_prefix=None,
                          figsize=(8, 6)):
    r"""
    Produce a simple figure showing the effect of deprojection on the (u, v)
    coordinates and visibilities

    Parameters
    ----------
    u, v : array, unit = :math:`\lambda`
        Projected (u, v) coordinates
    vis : array, unit = Jy
        Projected visibilities (complex: real + imag * 1j)
    weights : array, unit = Jy^-2
        Weights assigned to observed visibilities, of the form
        :math:`1 / \sigma^2`
    geom : SourceGeometry object
        Fitted geometry (see frank.geometry.SourceGeometry)
    bin_widths : list, unit = \lambda
        Bin widths in which to bin the observed visibilities
    logy : bool, default = False
        Whether to plot the visibility distributions in log(flux)
    force_style: bool, default = True
        Whether to use preconfigured matplotlib rcParams in generated figure
    save_prefix : string, default = None
        Prefix for saved figure name. If None, the figure won't be saved
    figsize : tuple = (width, height) of figure, unit = inch

    Returns
    -------
    fig : Matplotlib `.Figure` instance
        The produced figure, including the GridSpec
    axes : Matplotlib `~.axes.Axes` class
        The axes of the produced figure
    """

    logging.info('    Making deprojection figure')

    # Apply the deprojection to the provided (u, v) coordinates
    # and visibility amplitudes
    up, vp, visp = geom.apply_correction(u, v, vis)

    re_vis = np.real(vis)
    re_visp = np.real(visp)

    bsp = np.hypot(up, vp)

    with frank_plotting_style_context_manager(force_style):
        gs = GridSpec(2, 1, hspace=0.2, top=0.97, left=0.07, right=0.97)
        fig = plt.figure(figsize=figsize)

        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        axes = [ax0, ax1]

        plot_deprojection_effect(u / 1e6, v / 1e6, up / 1e6, vp / 1e6,
                                 re_vis * 1e3, re_visp * 1e3, axes)

    for i in range(len(bin_widths)):
        binned_vis = UVDataBinner(bsp, visp, weights, bin_widths[i])
        vis_re = binned_vis.V.real
        vis_err_re = binned_vis.error.real

        if logy:
            lab=r'Deprojected, >0, {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3)
        else:
            lab=r'Deprojected, {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3)
        plot_vis_quantity(binned_vis.uv / 1e6, vis_re * 1e3, ax1, c=cs[i],
             marker=ms[i], ls='None',
             label=lab)

        if logy:
            plot_vis_quantity(binned_vis.uv / 1e6, -vis_re * 1e3, ax1, c=cs2[i],
                 marker=ms[i], ls='None',
                 label=r'Deprojected, <0, {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))

    ax0.set_xlabel(r'u [M$\lambda$]')
    ax0.set_ylabel(r'v [M$\lambda$]')

    ax1.set_xlabel(r'Baseline [M$\lambda$]')
    ax1.set_ylabel('Re(V) [mJy]')
    if logx:
        ax1.set_xscale('log')
    if logy:
        ax1.set_yscale('log')
        ax1.set_ylim(bottom=1e-3)

    ax0.legend(loc=0)
    ax1.legend(loc=0)

    if save_prefix:
        plt.savefig(save_prefix + '_frank_deprojection.png', dpi=300,
                    bbox_inches='tight')
        plt.close()

    return fig, axes


def make_quick_fig(u, v, vis, weights, sol, bin_widths, priors, dist=None,
                   logx=False, force_style=True, save_prefix=None,
                   stretch='power', gamma=1.0, asinh_a=0.02, figsize=(8,6)):
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
    priors : dict
        Dictionary with fit hyperparameters: 'alpha', 'wsmooth', 'Rmax', 'N', 'p0'.
        Used for figure title
    dist : float, optional, unit = AU, default = None
        Distance to source, used to show second x-axis in [AU]
    logx : bool, default = False
        Whether to plot the visibility distributions in log(baseline)
    gamma : float, default = 1.0
        Index of power law normalization to apply to swept profile image's
        colormap (see matplotlib.colors.PowerNorm).
        gamma=1.0 yields a linear colormap
    force_style: bool, default = True
        Whether to use preconfigured matplotlib rcParams in generated figure
    save_prefix : string, default = None
        Prefix for saved figure name. If None, the figure won't be saved
    stretch : string, default = 'power'
        Transformation to apply to the colorscale. The default 'power' is a
        power law stretch. The other option is 'asinh', an arcsinh stretch,
        which requires astropy.visualization.mpl_normalize.simple_norm
    asinh_a : float, default = 0.02
        Scale parameter for an asinh stretch
    figsize : tuple = (width, height) of figure, unit = inch

    Returns
    -------
    fig : Matplotlib `.Figure` instance
        The produced figure, including the GridSpec
    axes : Matplotlib `~.axes.Axes` class
        The axes of the produced figure
    """

    logging.info('    Making quick figure')

    alpha, wsmooth, Rmax, N, p0 = priors['alpha'], priors['wsmooth'],\
                                  priors['Rmax'], priors['N'], priors['p0']

    with frank_plotting_style_context_manager(force_style):
        gs = GridSpec(3, 2, hspace=0, bottom=.12)
        gs2 = GridSpec(3, 2, hspace=.2)
        fig = plt.figure(figsize=figsize)
        fig.suptitle(r'Fit hyperparameters: $\alpha$={:.2f}, $w_{{smooth}}$={:.1e}, R$_{{max}}$={}, '\
                     'N={}, p$_0$={:.0e}'.format(alpha, wsmooth, Rmax, N, p0))

        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[2])

        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[3])

        ax4 = fig.add_subplot(gs2[4])
        ax5 = fig.add_subplot(gs2[5])

        ax0.text(.5, .9, 'a)', transform=ax0.transAxes)
        ax1.text(.5, .9, 'b)', transform=ax1.transAxes)

        ax2.text(.5, .9, 'c)', transform=ax2.transAxes)
        ax3.text(.5, .9, 'd)', transform=ax3.transAxes)

        ax4.text(.5, .9, 'e)', c='w', transform=ax4.transAxes)
        ax5.text(.5, .9, 'f)', c='w', transform=ax5.transAxes)

        axes = [ax0, ax1, ax2, ax3, ax4, ax5]

        plot_brightness_profile(sol.r, sol.I / 1e10, ax0, c='r', label='frank')

        if dist:
            ax0_5 = plot_brightness_profile(sol.r, sol.I / 1e10, ax0, dist=dist, c='r')
            xlims = ax0.get_xlim()
            ax0_5.set_xlim(np.multiply(xlims, dist))

        plot_brightness_profile(sol.r, sol.I / 1e10, ax1, c='r', label='frank')

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

            for ax in [ax2, ax3]:
                plot_vis_quantity(binned_vis.uv / 1e6, vis_re_kl, ax,
                     vis_err_re_kl, c=cs[i],
                     marker=ms[i], ls='None',
                     label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))

        vis_fit_kl = sol.predict_deprojected(grid).real * 1e3
        plot_vis_quantity(grid / 1e6, vis_fit_kl, ax2, c='r', label='frank', zorder=10)

        # Make a guess of good y-bounds for zooming in on the visibility fit
        # in linear-y
        zoom_ylim_guess = abs(vis_fit_kl[int(.5 * len(vis_fit_kl)):]).max()
        zoom_bounds = [-1.1 * zoom_ylim_guess, 1.1 * zoom_ylim_guess]
        ax3.set_ylim(zoom_bounds)

        plot_vis_quantity(grid / 1e6, vis_fit_kl, ax3, c='r', label='frank', zorder=10)

        vmax = sol.I.max()
        if stretch == 'asinh':
            vmin = max(0, min(sol.I))
            from astropy.visualization.mpl_normalize import simple_norm
            norm = simple_norm(sol.I, stretch='asinh', asinh_a=asinh_a, 
                               min_cut=vmin, max_cut=vmax)
        elif stretch == 'power':
            vmin = 0
            norm = PowerNorm(gamma, vmin, vmax)
        else:
            err = ValueError("Unknown 'stretch'. Should be one of 'power' or 'asinh'")
            raise err

        plot_2dsweep(sol.r, sol.I, ax=ax4, cmap='inferno', norm=norm,
                    project=False)
        plot_2dsweep(sol.r, sol.I, ax=ax5, cmap='inferno', norm=norm,
                    project=True, geom=sol.geometry)

        ax1.set_xlabel('r ["]')
        ax0.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
        ax1.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
        ax1.set_yscale('log')
        ax1.set_ylim(bottom=1e-3)

        ax3.set_xlabel(r'Baseline [M$\lambda$]')
        ax2.set_ylabel('Re(V) [mJy]')
        ax3.set_ylabel('Re(V) [mJy]')

        if logx:
            ax2.set_xscale('log')
            ax3.set_xscale('log')

            ax2.set_xlim(right=max(baselines) / 1e6 * 1.2)
        else:
            ax2.set_xlim(0, max(baselines) / 1e6 * 1.2)
        xlims = ax2.get_xlim()
        ax3.set_xlim(xlims)

        ax4.set_xlabel('RA offset ["]')
        ax4.set_ylabel('Dec offset ["]')
        ax5.set_xlabel('RA offset ["]')
        ax5.set_ylabel('Dec offset ["]')

        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)

        if save_prefix:
            plt.savefig(save_prefix + '_frank_fit_quick.png', dpi=300,
                        bbox_inches='tight')
            plt.close()

    return fig, axes


def make_full_fig(u, v, vis, weights, sol, bin_widths, priors,
                  dist=None, logx=False, force_style=True,
                  save_prefix=None, norm_residuals=False, stretch='power',
                  gamma=1.0, asinh_a=0.02, figsize=(8, 6)):
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
        Reconstructed profile using maximum a posteriori power spectrum
        (see frank.radial_fitters.FrankFitter)
    bin_widths : list, unit = \lambda
        Bin widths in which to bin the observed visibilities
    priors : dict
        Dictionary with fit hyperparameters: 'alpha', 'wsmooth', 'Rmax', 'N', 'p0'.
        Used for figure title
    dist : float, optional, unit = AU, default = None
        Distance to source, used to show second x-axis in [AU]
    logx : bool, default = False
        Whether to plot the visibility distributions in log(baseline)
    force_style: bool, default = True
        Whether to use preconfigured matplotlib rcParams in generated figure
    save_prefix : string, default = None
        Prefix for saved figure name. If None, the figure won't be saved
    norm_residuals : bool, default = False
        Whether to normalize the residual visibilities by the data's
        visibility amplitudes
    stretch : string, default = 'power'
        Transformation to apply to the colorscale. The default 'power' is a
        power law stretch. The other option is 'asinh', an arcsinh stretch,
        which requires astropy.visualization.mpl_normalize.simple_norm
    gamma : float, default = 1.0
        Index of power law normalization to apply to swept profile image's
        colormap (see matplotlib.colors.PowerNorm).
        gamma=1.0 yields a linear colormap
    asinh_a : float, default = 0.02
        Scale parameter for an asinh stretch
    figsize : tuple = (width, height) of figure, unit = inch

    Returns
    -------
    fig : Matplotlib `.Figure` instance
        The produced figure, including the GridSpec
    axes : Matplotlib `~.axes.Axes` class
        The axes of the produced figure
    """

    logging.info('    Making full figure')

    alpha, wsmooth, Rmax, N, p0 = priors['alpha'], priors['wsmooth'],\
                                  priors['Rmax'], priors['N'], priors['p0']

    with frank_plotting_style_context_manager(force_style):
        gs = GridSpec(3, 3, hspace=0, wspace=0.25, left=0.06, right=0.98)
        gs1 = GridSpec(4, 3, hspace=0, wspace=0.25, left=0.06, right=0.98)
        gs2 = GridSpec(3, 3, hspace=.35, wspace=0.25, left=0.06, right=0.98)
        fig = plt.figure(figsize=figsize)
        fig.suptitle(r'Fit hyperparameters: $\alpha$={:.2f}, $w_{{smooth}}$={:.1e}, R$_{{max}}$={}, '\
                     'N={}, p$_0$={:.0e}'.format(alpha, wsmooth, Rmax, N, p0))

        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[3])
        ax2 = fig.add_subplot(gs2[6])

        ax3 = fig.add_subplot(gs[1])
        ax4 = fig.add_subplot(gs[4])
        ax5 = fig.add_subplot(gs[7])

        ax6 = fig.add_subplot(gs1[2])
        ax7 = fig.add_subplot(gs1[5])
        ax8 = fig.add_subplot(gs1[8])
        ax9 = fig.add_subplot(gs1[11])

        ax0.text(.9, .6, 'a)', transform=ax0.transAxes)
        ax1.text(.9, .6, 'b)', transform=ax1.transAxes)
        ax2.text(.1, .9, 'c)', c='w', transform=ax2.transAxes)

        ax3.text(.9, .5, 'd)', transform=ax3.transAxes)
        ax4.text(.9, .1, 'e)', transform=ax4.transAxes)
        ax5.text(.9, .1, 'f)', transform=ax5.transAxes)
        ax6.text(.9, .1, 'g)', transform=ax6.transAxes)
        ax7.text(.9, .9, 'h)', transform=ax7.transAxes)
        ax8.text(.9, .1, 'i)', transform=ax8.transAxes)
        ax9.text(.9, .1, 'j)', transform=ax9.transAxes)

        axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

        # Plot the fitted brightness profile in linear- and log-y
        plot_brightness_profile(sol.r, sol.I / 1e10, ax0, c='r', label='frank')

        if dist:
            ax0_5 = plot_brightness_profile(sol.r, sol.I / 1e10, ax0, dist=dist, c='r')
            xlims = ax0.get_xlim()
            ax0_5.set_xlim(np.multiply(xlims, dist))

        plot_brightness_profile(sol.r, sol.I / 1e10, ax1, c='r', label='frank')

        # Apply deprojection to the provided (u, v) coordinates
        # and visibility amplitudes
        u_deproj, v_deproj, vis_deproj = sol.geometry.apply_correction(u, v, vis)
        baselines = (u_deproj**2 + v_deproj**2)**.5
        # Set a grid of baselines on which to plot the visibility domain frank fit
        grid = np.logspace(np.log10(min(baselines.min(), sol.q[0])),
                           np.log10(max(baselines.max(), sol.q[-1])), 10**4
                           )
        # Map the frank visibility fit to `grid`, considering only the real component
        # that frank fits
        vis_fit_kl = sol.predict_deprojected(grid).real * 1e3

        # Make a guess of good y-bounds for zooming in on the visibility fit
        # in linear-y
        zoom_ylim_guess = abs(vis_fit_kl[int(.5 * len(vis_fit_kl)):]).max()
        zoom_bounds = [-1.1 * zoom_ylim_guess, 1.1 * zoom_ylim_guess]
        ax4.set_ylim(zoom_bounds)

        # Bin the observed (real and imaginary components of the) visibilities
        # for plotting
        for i in range(len(bin_widths)):
            binned_vis = UVDataBinner(baselines, vis_deproj, weights, bin_widths[i])
            vis_re_kl = binned_vis.V.real * 1e3
            vis_im_kl = binned_vis.V.imag * 1e3
            vis_err_re_kl = binned_vis.error.real * 1e3
            vis_err_im_kl = binned_vis.error.imag * 1e3
            vis_fit = sol.predict_deprojected(binned_vis.uv).real * 1e3

            # Determine the visiblity domain frank fit residuals (and RMS error)
            # for Real(V)
            resid = vis_re_kl - vis_fit
            if norm_residuals:
                resid /= vis_re_kl
            rmse = (np.mean(resid**2))**.5

            # Plot the observed, binned visibilities (with errorbars) and the residuals
            plot_vis_quantity(binned_vis.uv / 1e6, vis_re_kl, ax3, c=cs[i],
                     marker=ms[i], ls='None',
                     label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))

            plot_vis_quantity(binned_vis.uv / 1e6, vis_re_kl, ax4, c=cs[i],
                     marker=ms[i], ls='None',
                     label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))

            plot_vis_quantity(binned_vis.uv / 1e6, vis_im_kl, ax6, c=cs[i],
                     marker=ms[i], ls='None',
                     label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))

            plot_vis_quantity(binned_vis.uv / 1e6, resid, ax5, c=cs[i], marker=ms[i], ls='None',
                           label=r'{:.0f} k$\lambda$ bins, RMSE {:.3f} mJy'.format(bin_widths[i]/1e3, rmse))

            # Plot a histogram of the observed visibilties to examine how the
            # visibility count varies with baseline
            plot_vis_hist(binned_vis, ax8, rescale=1e6, color=hist_cs[i],
                          label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))
            # Plot the binned data signal-to-noise as a function of baseline
            plot_vis_quantity(binned_vis.uv / 1e6,
                          binned_vis.V.real**2 / binned_vis.error.real**2,
                          ax9, color=hist_cs[i], marker=ms[i], ls='None',
                          label=r'{:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))

        # Plot the visibility domain frank fit in log-y
        plot_vis_quantity(grid / 1e6, vis_fit_kl, ax3, c='r', label='frank')
        plot_vis_quantity(grid / 1e6, vis_fit_kl, ax4, c='r', label='frank')

        # Plot the frank inferred power spectrum
        plot_vis_quantity(sol.q / 1e6, sol.power_spectrum, ax7)

        # Plot a sweep over 2\pi of the frank 1D fit
        # (analogous to a model image of the source)
        vmax = sol.I.max()
        if stretch == 'asinh':
            vmin = max(0, min(sol.I))
            from astropy.visualization.mpl_normalize import simple_norm
            norm = simple_norm(sol.I, stretch='asinh', asinh_a=asinh_a, 
                               min_cut=vmin, max_cut=vmax)
        elif stretch == 'power':
            vmin = 0
            norm = PowerNorm(gamma, vmin, vmax)
        else:
            err = ValueError("Unknown 'stretch'. Should be one of 'power' or 'asinh'")
            raise err

        plot_2dsweep(sol.r, sol.I, ax=ax2, cmap='inferno', norm=norm,
                    project=True, geom=sol.geometry)

        ax1.set_xlabel('r ["]')
        ax0.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
        ax1.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
        ax1.set_yscale('log')
        ax1.set_ylim(bottom=1e-3)
        ax2.set_xlabel('RA offset ["]')
        ax2.set_ylabel('Dec offset ["]')

        axs = [ax3, ax4, ax5, ax6, ax7, ax8, ax9]
        if logx:
            for aa in axs:
                aa.set_xscale('log')
            ax3.set_xlim(right=max(baselines) / 1e6 * 1.2)
        else:
            ax3.set_xlim(0, max(baselines) / 1e6 * 1.2)
        xlims = ax3.get_xlim()
        for aa in axs:
            aa.set_xlim(xlims)

        ax7.set_yscale('log')
        ax8.set_yscale('log')

        ax8.set_yticks([1, 10, 1e2, 1e3])

        ax9.set_yscale('log')
        ax9.set_ylim(1e-2,200)
        ax9.set_yticks([1e-2, 1e-1, 1, 10, 100])
        ax9.axhline(y=1, ls='--', c='c')

        ax3.set_ylabel('Re(V) [mJy]')
        ax4.set_ylabel('Re(V) [mJy]')
        if norm_residuals:
            ax5.set_ylabel('Norm. residual')
        else:
            ax5.set_ylabel('Residual [mJy]')
        ax5.set_xlabel(r'Baseline [M$\lambda$]')

        ax6.set_ylabel('Im(V) [mJy]')
        ax7.set_ylabel(r'Power [Jy$^2$]')
        ax8.set_ylabel('Count')
        ax9.set_ylabel(r'SNR=$\mu_{\rm bin}^2 / \sigma_{\rm bin}^2$')
        ax9.set_xlabel(r'Baseline [M$\lambda$]')

        axs = [ax0, ax3, ax4, ax6, ax7, ax8]
        for aa in axs:
            plt.setp(aa.get_xticklabels(), visible=False)

        if save_prefix:
            plt.savefig(save_prefix + '_frank_fit_full.png', dpi=300)
            plt.close()

    return fig, axes


def make_diag_fig(r, q, iteration_diagnostics, iter_plot_range=None, logx=False,
                  force_style=True, save_prefix=None, figsize=(8, 6)):
    r"""
    Produce a diagnostic figure showing fit convergence metrics

    Parameters
    ----------
    r : array
        Radial data coordinates at which the brightness profile is defined.
        The assumed unit (for the x-label) is arcsec
    q : array
        Baselines at which the power spectrum is defined.
        The assumed unit (for the x-label) is :math:`\lambda`
    iteration_diagnostics : dict
        The iteration_diagnostics from FrankFitter.
    iter_plot_range : list
        Range of iterations in the fit over which to
        plot brightness profile and power spectrum reconstructions
    logx : bool, default = False
        Whether to plot the visibility distributions in log(baseline)
    force_style: bool, default = True
        Whether to use preconfigured matplotlib rcParams in generated figure
    save_prefix : string, default = None
        Prefix for saved figure name. If None, the figure won't be saved
    figsize : tuple = (width, height) of figure, unit = inch

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

    with frank_plotting_style_context_manager(force_style):
        gs = GridSpec(2, 2, hspace=0, bottom=.35)
        gs2 = GridSpec(3, 2, hspace=0, top=.7)
        fig = plt.figure(figsize=figsize)

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

        profile_iter = iteration_diagnostics['MAP']
        profile_iter_toplot = [x / 1e10 for x in profile_iter]
        pwr_spec_iter = iteration_diagnostics['power_spectrum']
        num_iter = iteration_diagnostics['num_iterations']

        plot_iterations(r, profile_iter_toplot, iter_plot_range, ax0)

        # Plot the difference in the profile between the last 100 iterations
        iter_plot_range_end = [max(iter_plot_range[1] - 100, 0),
                               iter_plot_range[1] - 1]

        plot_iterations(r, np.diff(profile_iter_toplot, axis=0),
                                iter_plot_range_end, ax1,
                                cmap=plt.cm.cividis)  # pylint: disable=no-member

        plot_iterations(q, pwr_spec_iter, iter_plot_range, ax2)

        # Plot the difference in the power spectrum between the last 100 iterations
        plot_iterations(q, abs(np.diff(pwr_spec_iter, axis=0)),
                                 iter_plot_range_end, ax3,
                                 cmap=plt.cm.cividis)  # pylint: disable=no-member

        plot_convergence_criterion(profile_iter_toplot, num_iter, ax4, c='k')

        ax0.set_ylabel(r'I [$10^{10}$ Jy sr$^{-1}$]')
        ax1.set_ylabel(r'$I_i - I_{i-1}$ [$10^{10}$ Jy sr$^{-1}$]')
        ax1.set_xlabel('r ["]')

        ax2.set_ylabel(r'Power [Jy$^2$]')
        ax3.set_ylabel(r'|PS$_i$ - PS$_{i-1}$| [Jy$^2$]')
        ax3.set_xlabel(r'Baseline [$\lambda$]')
        if logx:
            ax2.set_xscale('log')
            ax3.set_xscale('log')
        ax2.set_yscale('log')
        ax3.set_yscale('log')

        ax4.set_xlabel('Fit iteration')
        ax4.set_ylabel('Convergence criterion,\n' +
                       r'max(|$I_i - I_{i-1}$|) / max($I_i$)')
        ax4.set_yscale('log')

        xlims = ax2.get_xlim()
        ax3.set_xlim(xlims)

        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)

        if save_prefix:
            plt.savefig(save_prefix + '_frank_fit_diag.png', dpi=300,
                        bbox_inches='tight')
            plt.close()

    return fig, axes, iter_plot_range


def make_clean_comparison_fig(u, v, vis, weights, sol, clean_profile,
                              bin_widths, stretch='power',
                              gamma=1.0, asinh_a=0.02, MAP_convolved=None,
                              dist=None, logx=False, logy=False, ylims=None,
                              force_style=True, save_prefix=None,
                              figsize=(8, 10)):
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
    clean_profile : dict
        Dictionary with entries 'r' for the radial points [arcsec],
        'I' for the brightness [Jy / sr], and optionally the negative and positive
        brightness uncertainties 'lo_err' and 'hi_err' [Jy / sr]. If only the
        negative uncertainty is provided, the positive uncertainty is assumed
        equal to it
    bin_widths : list, unit = \lambda
        Bin widths in which to bin the observed visibilities
    stretch : string, default = 'power'
        Transformation to apply to the colorscale. The default 'power' is a
        power law stretch. The other option is 'asinh', an arcsinh stretch,
        which requires astropy.visualization.mpl_normalize.simple_norm
    gamma : float, default = 1.0
        Index of power law normalization to apply to swept profile image's
        colormap (see matplotlib.colors.PowerNorm).
        gamma=1.0 yields a linear colormap
    asinh_a : float, default = 0.02
        Scale parameter for an asinh stretch
    MAP_convolved : None (default) or array, unit = Jy / sr
        frank brightness profile convolved with a CLEAN beam
        (see utilities.convolve_profile).
        The assumed unit is for the x-label
    dist : float, optional, unit = AU, default = None
        Distance to source, used to show second x-axis in [AU]
    logx : bool, default = False
        Whether to plot the visibility distributions in log(baseline)
    logy : bool, default = False
        Whether to plot the visibility distributions in log(flux)
    ylims : list of len(2), default = None
        Lower and upper y-bounds for the visibility domain plot
    force_style: bool, default = True
        Whether to use preconfigured matplotlib rcParams in generated figure
    save_prefix : string, default = None
        Prefix for saved figure name. If None, the figure won't be saved
    figsize : tuple = (width, height) of figure, unit = inch

    Returns
    -------
    fig : Matplotlib `.Figure` instance
        The produced figure, including the GridSpec
    axes : Matplotlib `~.axes.Axes` class
        The axes of the produced figure
    """

    logging.info('    Making CLEAN comparison figure')

    with frank_plotting_style_context_manager(force_style):
        gs = GridSpec(3, 1)
        gs2 = GridSpec(3, 3)

        fig = plt.figure(figsize=figsize)

        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])

        ax2 = fig.add_subplot(gs2[6])
        ax3 = fig.add_subplot(gs2[7])
        ax4 = fig.add_subplot(gs2[8])

        axes = [ax0, ax1, ax2, ax3, ax4]

        if 'lo_err' in clean_profile.keys():
            low_uncer = clean_profile['lo_err']
        else:
            low_uncer = None
        if 'hi_err' in clean_profile.keys():
            high_uncer = clean_profile['hi_err']
        else:
            high_uncer = None

        plot_brightness_profile(clean_profile['r'], clean_profile['I'] / 1e10, ax0,
                                low_uncer=low_uncer, high_uncer=high_uncer,
                                c='b', ls='--', label='CLEAN')

        plot_brightness_profile(sol.r, sol.I / 1e10, ax0, c='r', ls=':', label='frank')

        if MAP_convolved is not None:
            plot_brightness_profile(sol.r, MAP_convolved / 1e10, ax0, c='k', ls='-',
                                    label='frank, convolved')

        if dist:
            ax0_5 = plot_brightness_profile(sol.r, sol.I / 1e10, ax0, dist=dist, c='r', ls=':')

        u_deproj, v_deproj, vis_deproj = sol.geometry.apply_correction(u, v, vis)
        baselines = (u_deproj**2 + v_deproj**2)**.5
        grid = np.logspace(np.log10(min(baselines.min(), sol.q[0])),
                           np.log10(max(baselines.max(), sol.q[-1])), 10**4
                           )

        for i in range(len(bin_widths)):
            binned_vis = UVDataBinner(baselines, vis_deproj, weights, bin_widths[i])
            vis_re_kl = binned_vis.V.real * 1e3
            vis_err_re_kl = binned_vis.error.real * 1e3

            if logy:
                lab=r'Obs.>0, {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3)
            else:
                lab=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3)
            plot_vis_quantity(binned_vis.uv / 1e6, vis_re_kl, ax1, c=cs[i],
                     marker=ms[i], ls='None',
                     label=lab)
            if logy:
                plot_vis_quantity(binned_vis.uv / 1e6, -vis_re_kl, ax1, c=cs2[i],
                     marker=ms[i], ls='None',
                     label=r'Obs.<0, {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))

        vis_fit_kl = sol.predict_deprojected(grid).real * 1e3

        # Take the discrete Hankel transform of the CLEAN profile, using the same
        # collocation points for the DHT as those in the frank fit
        Inu_interp = np.interp(
            sol.r, clean_profile['r'], clean_profile['I'].real) * 1e3
        clean_DHT_kl = sol.predict_deprojected(grid, I=Inu_interp)

        if logy:
            lab = 'frank, >0'
            lab2 = 'DHT of CLEAN, >0'
        else:
            lab = 'frank'
            lab2 = 'DHT of CLEAN'
        plot_vis_quantity(grid / 1e6, vis_fit_kl, ax1, c='r', label=lab)
        plot_vis_quantity(grid / 1e6, clean_DHT_kl, ax1, c='b', label=lab2)
        if logy:
            plot_vis_quantity(grid / 1e6, -vis_fit_kl, ax1, c='r', ls='--', label='frank, <0')
            plot_vis_quantity(grid / 1e6, -clean_DHT_kl, ax1, c='b', ls='--', label='DHT of CLEAN, <0')

        if logx:
            ax1.set_xscale('log')
        ax1.set_xlim(right=1.2 * max(baselines) / 1e6)

        if logy:
            ax1.set_yscale('log')
            ax1.set_ylim(bottom=1e-3)
        if ylims:
            ax1.set_ylim(ylims)

        if MAP_convolved is not None:
            vmax = max(sol.I.max(), MAP_convolved.max(), clean_profile['I'].max())
        else:
            vmax = max(sol.I.max(), clean_profile['I'].max())
        if stretch == 'asinh':
            vmin = max(0, min(sol.I))
            from astropy.visualization.mpl_normalize import simple_norm
            norm = simple_norm(sol.I, stretch='asinh', asinh_a=asinh_a, 
                               min_cut=vmin, max_cut=vmax)
        elif stretch == 'power':
            vmin = 0
            norm = PowerNorm(gamma, vmin, vmax)
        else:
            err = ValueError("Unknown 'stretch'. Should be one of 'power' or 'asinh'")
            raise err

        plot_2dsweep(sol.r, sol.I, ax=ax2, cmap='inferno', norm=norm,
                    xmax=sol.Rmax, plot_colorbar=True)
        if MAP_convolved is not None:
            plot_2dsweep(sol.r, MAP_convolved, ax=ax3, cmap='inferno', norm=norm,
                        xmax=sol.Rmax, plot_colorbar=True)

        # Interpolate the CLEAN profile onto the frank grid to ensure the CLEAN
        # swept 'image' has the same pixel resolution as the frank swept 'images'
        from scipy.interpolate import interp1d
        interp = interp1d(clean_profile['r'], clean_profile['I'])
        regrid_I_clean = interp(sol.r)
        plot_2dsweep(sol.r, regrid_I_clean, ax=ax4, cmap='inferno', norm=norm,
                    xmax=sol.Rmax, plot_colorbar=True)

        ax0.legend(loc='best')
        ax1.legend(loc='best', ncol=2)

        ax0.set_xlabel('r ["]')
        ax0.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
        ax1.set_xlabel(r'Baseline [M$\lambda$]')
        ax1.set_ylabel(r'Re(V) [mJy]')
        ax2.set_xlabel('RA offset ["]')
        ax3.set_xlabel('RA offset ["]')
        ax4.set_xlabel('RA offset ["]')
        ax2.set_ylabel('Dec offset ["]')

        ax0.set_xlim(right=sol.Rmax)
        if dist:
            xlims = ax0.get_xlim()
            ax0_5.set_xlim(np.multiply(xlims, dist))

        if logx:
            ax1.set_xscale('log')
        if logy:
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
            plt.savefig(save_prefix + '_frank_clean_comparison.png', dpi=300)
            plt.close()

    return fig, axes


def make_multifit_fig(u, v, vis, weights, sols, bin_widths, dist=None,
                      logx=False, force_style=True,
                      save_prefix=None, figsize=(8, 8)
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
    sols : list of _HankelRegressor objects
        Reconstructed profile using Maximum a posteriori power spectrum
        (see frank.radial_fitters.FrankFitter), for each of multiple fits
    bin_widths : list, unit = \lambda
        Bin widths in which to bin the observed visibilities
    dist : float, optional, unit = AU, default = None
        Distance to source, used to show second x-axis in [AU]
    logx : bool, default = False
        Whether to plot the visibility distributions in log(baseline)
    force_style: bool, default = True
        Whether to use preconfigured matplotlib rcParams in generated figure
    save_prefix : string, default = None
        Prefix for saved figure name. If None, the figure won't be saved
    figsize : tuple = (width, height) of figure, unit = inch

    Returns
    -------
    fig : Matplotlib `.Figure` instance
        The produced figure, including the GridSpec
    axes : Matplotlib `~.axes.Axes` class
        The axes of the produced figure
    """

    logging.info('    Making multifit figure')

    with frank_plotting_style_context_manager(force_style):
        gs = GridSpec(3, 2, hspace=0)
        fig = plt.figure(figsize=figsize)

        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[2])

        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[3])
        ax4 = fig.add_subplot(gs[5])

        ax0.text(.9, .4, 'a)', transform=ax0.transAxes)
        ax1.text(.5, .9, 'b)', transform=ax1.transAxes)

        ax2.text(.5, .9, 'c)', transform=ax2.transAxes)
        ax3.text(.5, .9, 'd)', transform=ax3.transAxes)
        ax4.text(.5, .9, 'e)', transform=ax4.transAxes)

        axes = [ax0, ax1, ax2, ax3, ax4]

        # Assume the fitted geometry and thus deprojected baseline distribution
        # is the same for all fits, plotting the common dataset
        u_deproj, v_deproj, vis_deproj = sols[0].geometry.apply_correction(u, v, vis)
        baselines = (u_deproj**2 + v_deproj**2)**.5
        grid = np.logspace(np.log10(min(baselines.min(), sols[0].q[0])),
                           np.log10(max(baselines.max(), sols[0].q[-1])),
                           10**4)

        for i in range(len(bin_widths)):
            binned_vis = UVDataBinner(
                baselines, vis_deproj, weights, bin_widths[i])
            vis_re_kl = binned_vis.V.real * 1e3
            vis_err_re_kl = binned_vis.error.real * 1e3

            plot_vis_quantity(binned_vis.uv, vis_re_kl, ax2, c=cs[i],
                     marker=ms[i], ls='None', label=r'Obs., {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))

            plot_vis_quantity(binned_vis.uv, vis_re_kl, ax3, c=cs[i],
                     marker=ms[i], ls='None',
                     label=r'Obs.>0, {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))
            plot_vis_quantity(binned_vis.uv, -vis_re_kl, ax3, c=cs2[i],
                     marker=ms[i], ls='None',
                     label=r'Obs.<0, {:.0f} k$\lambda$ bins'.format(bin_widths[i]/1e3))

        # Overplot the multiple fits
        for ii in range(len(sols)):
            plot_brightness_profile(sols[ii].r, sols[ii].I / 1e10, ax0, c=multifit_cs[ii])#,
                # label='alpha = {}, wsmooth = {}'.format(sols[ii].info['alpha'], sols[ii].info['wsmooth'])
            if dist and ii == len(sols) - 1:
                ax0_5 = plot_brightness_profile(sols[ii].r, sols[ii].I / 1e10, ax0, dist=dist, c=multifit_cs[ii])
                xlims = ax0.get_xlim()
                ax0_5.set_xlim(np.multiply(xlims, dist))

            plot_brightness_profile(sols[ii].r, sols[ii].I / 1e10, ax1, c=multifit_cs[ii])

            vis_fit_kl = sols[ii].predict_deprojected(grid).real * 1e3
            plot_vis_quantity(grid, vis_fit_kl, ax2, c=multifit_cs[ii])

            plot_vis_quantity(grid, vis_fit_kl, ax3, c=multifit_cs[ii])
            plot_vis_quantity(grid, -vis_fit_kl, ax3, c=multifit_cs[ii], ls='--')

            plot_vis_quantity(sols[ii].q, sols[ii].power_spectrum, ax4, c=multifit_cs[ii])

        ax1.set_xlabel('r ["]')
        ax0.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
        ax1.set_ylabel(r'Brightness [$10^{10}$ Jy sr$^{-1}$]')
        ax1.set_yscale('log')
        ax1.set_ylim(bottom=1e-3)

        ax4.set_xlabel(r'Baseline [$\lambda$]')
        ax2.set_ylabel('Re(V) [mJy]')
        ax3.set_ylabel('Re(V) [mJy]')
        ax4.set_ylabel(r'Power [Jy$^2$]')
        if logx:
            ax2.set_xscale('log')
            ax3.set_xscale('log')
            ax4.set_xscale('log')
        ax3.set_yscale('log')
        ax4.set_yscale('log')
        ax3.set_ylim(bottom=1e-4)

        xlims = ax2.get_xlim()
        ax3.set_xlim(xlims)
        ax4.set_xlim(xlims)

        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)

        ax0.legend()
        ax2.legend()
        ax3.legend()

        if save_prefix:
            plt.savefig(save_prefix + '_frank_multifit.png', dpi=300)
            plt.close()

    return fig, axes


def make_bootstrap_fig(r, profiles, force_style=True,
                       save_prefix=None, figsize=(8, 6)):
    r"""
    Produce a figure showing a bootstrap analysis for a Frankenstein fit

    Parameters
    ----------
    r : array, unit = arcsec
        Single set of radial collocation points used in all bootstrap fits
    profiles : array, unit = Jy / sr
        Brightness profiles of all bootstrap fits
    force_style: bool, default = True
        Whether to use preconfigured matplotlib rcParams in generated figure
    save_prefix : string, default = None
        Prefix for saved figure name. If None, the figure won't be saved
    figsize : tuple = (width, height) of figure, unit = inch

    Returns
    -------
    fig : Matplotlib `.Figure` instance
        The produced figure, including the GridSpec
    axes : Matplotlib `~.axes.Axes` class
        The axes of the produced figure
    """

    logging.info('    Making bootstrap summary figure')

    with frank_plotting_style_context_manager(force_style):
        gs = GridSpec(2, 2, hspace=0)
        fig = plt.figure(figsize=figsize)

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

        plot_brightness_profile(r, mean_profile / 1e10, ax1, low_uncer=std / 1e10,
                                 color='r', alpha=.7, label='Stan. dev. of trials')
        plot_brightness_profile(r, mean_profile / 1e10, ax3, low_uncer=std / 1e10,
                                 color='r', alpha=.7, label='Stan. dev. of trials')

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

        if save_prefix:
            plt.savefig(save_prefix + '_frank_bootstrap.png', dpi=300,
                        bbox_inches='tight')
            plt.close()

    return fig, axes
