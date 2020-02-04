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

import numpy as np

def load_uvtable(data_file):
    """
    Read in a UVTable with data to be fit.

    Parameters
    ----------
    data_file : string
          UVTable with columns: u [lambda]  v [lambda]  Re(V) [Jy]  Im(V) [Jy]
                                Weight [Jy^-2]

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


def save_fit(model, u, v, vis, weights, sol):
    """
    Save datafiles of fit results

    Parameters
    ----------
    model : dict
          Dictionary containing model parameters the fit uses
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
    """

    prefix = model['input_output']['save_dir'] + '/' + \
             os.path.splitext(model['input_output']['uvtable_filename'])[0]

    if model['input_output']['save_profile_fit']:
        np.savetxt(prefix + '_frank_profile_fit.txt',
                   np.array([sol.r, sol.mean, np.diag(sol.covariance)**.5]).T,
                   header='r [arcsec]\tI [Jy/sr]\tI_uncer [Jy/sr]')

    if model['input_output']['save_vis_fit']:
        np.savetxt(prefix + '_fit_vis.txt',
                   np.array([sol.q, sol.predict_deprojected(sol.q).real]).T,
                   header='Baseline [lambda]\tProjected Re(V) [Jy]') # TODO: update 

    if model['input_output']['save_uvtables']:
        np.savetxt(prefix + '_frank_uv_fit.txt',
                np.stack([u, v, sol.predict(u,v).real, sol.predict(u,v).imag,
                weights], axis=-1), header='u [lambda]\tv [lambda]\tRe(V)'
                ' [Jy]\tIm(V) [Jy]\tWeight [Jy^-2]')
        np.savetxt(prefix + '_frank_uv_resid.txt',
                np.stack([u, v, vis.real - sol.predict(u,v).real,
                vis.imag - sol.predict(u,v).imag, weights], axis=-1),
                header='u [lambda]\tv [lambda]\tRe(V) [Jy]\tIm(V) [Jy]\tWeight'
                ' [Jy^-2]')
