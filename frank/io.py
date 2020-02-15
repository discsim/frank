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
"""This module has functions that read in datafiles and save fit results to
   file.
"""

import os
import numpy as np
import json

def load_uvtable(data_file):
    """
    Read in a UVTable with data to be fit.

    Parameters
    ----------
    data_file : string
          UVTable with columns: u [lambda]  v [lambda]  Re(V) [Jy]  Im(V) [Jy]
                                Weight [Jy^-2] TODO: update if accept dfft struc

    Returns
    -------
    u, v : array, unit = :math:`\\lambda`
          u and v coordinates of observations
    vis : array, unit = Jy
          Real component of observed visibilities
    weights : array, unit = Jy^-2
          Weights assigned to observed visibilities, of the form
          :math:`1 / \\sigma^2`
    """

    u, v, vis, weights = np.genfromtxt(data_file).T
    # TODO: add other file types to accept (.npy, .npz)
    # TODO: allow other column orders in UVTable
    # TODO: (optionally) convert u, v from [m] to [lambda]?

    return u, v, vis, weights


def save_fit(u, v, vis, weights, sol, save_dir, uvtable_filename,
             save_profile_fit, save_vis_fit, save_uvtables):
    """
    Save datafiles of fit results

    Parameters
    ----------
    u, v : array, unit = :math:`\\lambda`
          u and v coordinates of observations
    vis : array, unit = Jy
          Real component of observed visibilities
    weights : array, unit = Jy^-2
          Weights assigned to observed visibilities, of the form
          :math:`1 / \\sigma^2`
    sol : _HankelRegressor object
          Reconstructed profile using Maximum a posteriori power spectrum
          (see frank.radial_fitters.FrankFitter)
    save_dir : string
          Directory in which output datafiles and figures are saved
    uvtable_filename : string
          UVTable with data to be fit. (columns: u, v, Re(V), Im(V), weights) # TODO
    save_profile_fit : bool
          Whether to save fitted brightness profile
    save_vis_fit : bool
          Whether to save fitted visibility distribution
    save_uvtables : bool
          Whether to save fitted and residual UV tables (these are reprojected)
    """

    prefix = save_dir + '/' + os.path.splitext(uvtable_filename)[0]

    with open(prefix + '_sol.json', 'w') as f:
        json.dump(model, f, indent=4)

    if save_profile_fit:
        np.savetxt(prefix + '_frank_profile_fit.txt',
                   np.array([sol.r, sol.mean, np.diag(sol.covariance)**.5]).T,
                   header='r [arcsec]\tI [Jy/sr]\tI_uncer [Jy/sr]')

    if save_vis_fit:
        np.savetxt(prefix + '_frank_vis_fit.txt',
                   np.array([sol.q, sol.predict_deprojected(sol.q).real]).T,
                   header='Baseline [lambda]\tProjected Re(V) [Jy]') # TODO: update

    if save_uvtables:
        np.savetxt(prefix + '_frank_uv_fit.txt',
                np.stack([u, v, sol.predict(u,v).real, sol.predict(u,v).imag,
                weights], axis=-1), header='u [lambda]\tv [lambda]\tRe(V)'
                ' [Jy]\tIm(V) [Jy]\tWeight [Jy^-2]')
        np.savetxt(prefix + '_frank_uv_resid.txt',
                np.stack([u, v, vis.real - sol.predict(u,v).real,
                vis.imag - sol.predict(u,v).imag, weights], axis=-1),
                header='u [lambda]\tv [lambda]\tRe(V) [Jy]\tIm(V) [Jy]\tWeight'
                ' [Jy^-2]')
