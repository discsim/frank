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
"""This module generates figures for a Frankenstein fit and/or its diagnostics.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from frank import plot, useful_funcs

#plt.style.use('paper')
def use_frank_plotting_style():
    """#TODO
    """
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


def make_full_fig(u, v, vis, weights, sol, save_dir, uvtable_filename, bin_widths, dist):
    prefix = save_dir + '/' + os.path.splitext(uvtable_filename)[0]

    gs = GridSpec(3, 3, hspace=0)
    gs2 = GridSpec(3, 3, hspace=.35)
    fig = plt.figure(figsize=(8,6))

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
                       np.log10(max(baselines.max(), sol.q[-1])),
                       10**4)

    zoom_ylim_guess = abs(sol.predict_deprojected(grid).real[np.int(.5*len(sol.predict_deprojected(grid).real)):]).max()
    zoom_bounds = [-1.1 * zoom_ylim_guess, 1.1 * zoom_ylim_guess]

    cs = ['#a4a4a4', 'k', '#896360', 'b']
    cs2 = ['#3498DB', 'm', '#F9B817', '#ED6EFF']
    ms = ['x', '+', '.', '1']

    for i in range(len(bin_widths)):
        binned_vis = BinUVData(baselines, vis_deproj, weights, bin_widths[i])
        vis_re_kl = binned_vis.V.real * 1e3
        vis_im_kl = binned_vis.V.imag * 1e3
        vis_err_re_kl = binned_vis.error.real * 1e3
        vis_err_im_kl = binned_vis.error.imag * 1e3

        plot_vis(binned_vis.uv, vis_re_kl,
            vis_err_re_kl, ax3, c=cs[i], marker=ms[i], binwidth=bin_widths[i])
        plot_vis(binned_vis.uv, vis_re_kl,
            vis_err_re_kl, ax4, c=cs[i], marker=ms[i], binwidth=bin_widths[i], zoom=np.multiply(zoom_bounds, 1e3))
        plot_vis(binned_vis.uv, vis_re_kl,
            vis_err_re_kl, ax6, c=cs[i], c2=cs2[i], marker=ms[i], marker2=ms[i], binwidth=bin_widths[i], yscale='log')

        plot_vis(binned_vis.uv, vis_im_kl,
            vis_err_im_kl, ax8, c=cs[i], marker=ms[i], binwidth=bin_widths[i], ylabel='Im(V) [mJy]')

        plot_vis_resid(binned_vis.uv, vis_re_kl,
            sol.predict_deprojected(binned_vis.uv).real * 1e3, ax5, c=cs[i], marker=ms[i], binwidth=bin_widths[i], normalize_resid=False)

    vis_fit_kl = sol.predict_deprojected(grid).real * 1e3
    plot_vis_fit(grid, vis_fit_kl, ax3)
    plot_vis_fit(grid, vis_fit_kl, ax4)
    plot_vis_fit(grid,vis_fit_kl, ax6, yscale='log', ylolim=1e-4, ls2='--')

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

    plt.savefig(prefix + '_frank_fit_full.png', dpi=600)

    return fig, axes

def make_quick_fig(u, v, vis, weights, sol, save_dir, uvtable_filename, bin_widths, dist):
    prefix = save_dir + '/' + os.path.splitext(uvtable_filename)[0]

    gs = GridSpec(2, 2, hspace=0)
    fig = plt.figure(figsize=(8,6))

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

    cs = ['#a4a4a4', 'k', '#896360', 'b']
    cs2 = ['#3498DB', 'm', '#F9B817', '#ED6EFF']
    ms = ['x', '+', '.', '1']

    for i in range(len(bin_widths)):
        binned_vis = BinUVData(baselines, vis_deproj, weights, bin_widths[i])
        vis_re_kl = binned_vis.V.real * 1e3
        vis_im_kl = binned_vis.V.imag * 1e3
        vis_err_re_kl = binned_vis.error.real * 1e3
        vis_err_im_kl = binned_vis.error.imag * 1e3

        plot_vis(binned_vis.uv, vis_re_kl,
            vis_err_re_kl, ax2, c=cs[i], marker=ms[i], binwidth=bin_widths[i])

        plot_vis_resid(binned_vis.uv, vis_re_kl,
            sol.predict_deprojected(binned_vis.uv).real * 1e3, ax3, c=cs[i], marker=ms[i], binwidth=bin_widths[i], normalize_resid=False)

    vis_fit_kl = sol.predict_deprojected(grid).real * 1e3
    plot_vis_fit(grid, vis_fit_kl, ax2)

    xlims = ax2.get_xlim()
    ax3.set_xlim(xlims)

    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.tight_layout()

    plt.savefig(prefix + '_frank_fit_quick.png', dpi=600)

    return fig, axes
