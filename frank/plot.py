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

def plot_fit(model, u, v, vis, weights, geom, sol, diag_fig=True,
             save_plots=True):
    """ # TODO: add docstring
    """

    from frank.constants import deg_to_rad
    plt.figure()
    plt.loglog(np.hypot(u,v), vis.real, 'k.')
    plt.loglog(np.hypot(u,v), sol.predict(u,v).real,'g.')
    plt.loglog(sol.q, sol.predict_deprojected(sol.q).real, 'r.')
    plt.savefig(prefix + '_test.png')
