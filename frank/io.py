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
"""This module has functions that read in a UVTable and save fit results to
   file.
"""

import os
import numpy as np
import pickle


def load_uvtable(data_file):
    r"""
    Read in a UVTable with data to be fit

    Parameters
    ----------
    data_file : string
          UVTable with data to be fit, with columns:
          u [lambda]  v [lambda]  Re(V) [Jy]  Im(V) [Jy] Weight [Jy^-2]

    Returns
    -------
    u, v : array, unit = :math:`\lambda`
          u and v coordinates of observations
    vis : array, unit = Jy
          Observed visibilities (complex: real + imag * 1j)
    weights : array, unit = Jy^-2
          Weights on the visibilities, of the form
          :math:`1 / \sigma^2`
    """
    import os.path
    extension = os.path.splitext(data_file)[1]

    if extension in {'.txt', '.dat'}:
        u, v, re, im, weights = np.genfromtxt(data_file).T
        vis = re + 1j*im

    elif extension == '.npz':
        dat = np.load(data_file)
        u, v, vis, weights = [dat[i] for i in ['u', 'v', 'V', 'weights']]

    else:
        raise ValueError("    You provided a UVTable with the extension %s."
                         " Please provide it as a `.txt`, `.dat`, `.npy`, or"
                         " `.npz`." % extension)

    return u, v, vis, weights


def save_fit(u, v, vis, weights, sol, prefix,
             save_profile_fit=True, save_vis_fit=True, save_uvtables=True,
             save_iteration_diag=False, format='npz',
             ):
    r"""
    Save datafiles of fit results

    Parameters
    ----------
    u, v : array, unit = :math:`\lambda`
        u and v coordinates of observations
    vis : array, unit = Jy
        Complex visibilities
    weights : array, unit = Jy^-2
        Weights on the visibilities, of the form
        :math:`1 / \sigma^2`
    sol : _HankelRegressor object
        Reconstructed profile using Maximum a posteriori power spectrum
        (see frank.radial_fitters.FrankFitter)
    prefix : string
        Base part of the filename
    save_profile_fit : bool
        Whether to save fitted brightness profile
    save_vis_fit : bool
        Whether to save fitted visibility distribution.
        NOTE: This is deprojected
    save_uvtables : bool
        Whether to save fitted and residual UV tables.
        NOTE: These are reprojected
    save_iteration_diag : bool
        Whether to save diagnostics of the fit iteration
    format : string, default=npz
        Format to save the uvtables in.
    """

    if not (format == 'txt' or format == 'dat' or format == 'npz'):
        raise ValueError("Format must be 'npz', 'txt', or 'dat'.")

    with open(prefix + '_frank_sol.obj', 'wb') as f:
        pickle.dump(sol, f)

    if save_iteration_diag:
        with open(prefix + '_frank_iteration_diagnostics.obj', 'wb') as f:
            pickle.dump(iteration_diagnostics, f)

    if save_profile_fit:
        np.savetxt(prefix + '_frank_profile_fit.txt',
                   np.array([sol.r, sol.mean, np.diag(sol.covariance)**.5]).T,
                   header='r [arcsec]\tI [Jy/sr]\tI_uncer [Jy/sr]')

    if save_vis_fit:
        if format == 'txt' or format == 'dat':
            np.savetxt(prefix + '_frank_vis_fit.' + format,
                       np.array([sol.q, sol.predict_deprojected(sol.q).real]).T,
                       header='Baseline [lambda]\tProjected Re(V) [Jy]')
        elif format == 'npz':
            np.savez(prefix + '_frank_vis_fit.' + format,
                     uv=sol.q, V=sol.sol.predict_deprojected(sol.q),
                     units={'uv': 'lambda', 'V': 'Jy'})

    if save_uvtables:
        V_pred = sol.predict(u, v)

        if format == 'txt' or format == 'dat':
            header = 'u [lambda]\tv [lambda]\tRe(V)  [Jy]\tIm(V) [Jy]\tWeight [Jy^-2]'

            np.savetxt(prefix + '_frank_uv_fit.' + format,
                       np.stack([u, v, V_pred.real, V_pred.imag,
                                 weights], axis=-1), header=header)

            np.savetxt(prefix + '_frank_uv_resid.' + format,
                       np.stack([u, v, vis.real - V_pred.real,
                                 vis.imag - V_pred.imag, weights], axis=-1),
                       header=header)

        elif format == 'npz':
            units = {'u': 'lambda', 'v': 'lambda',
                     'V': 'Jy', 'weights': "Jy^-2"}
            np.savez(prefix + '_frank_vis_fit.' + format,
                     u=u, v=v, V=V_pred, weights=weights,
                     units=units)

            np.savez(prefix + '_frank_vis_resid.' + format,
                     u=u, v=v, V=vis - V_pred, weights=weights,
                     units=units)
