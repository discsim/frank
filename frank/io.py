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
import logging

def load_uvtable(data_file):
    r"""
    Read in a UVTable with data to be fit

    Parameters
    ----------
    data_file : string
          UVTable with data to be fit.
          If in ASCII format, the table should have columns:
            u [lambda]  v [lambda]  Re(V) [Jy]  Im(V) [Jy]  Weight [Jy^-2]
          If in .npz format, the file should have arrays:
            "u" [lambda], "v" [lambda], "V" [Jy; complex: real + imag * 1j],
            "weights" [Jy^-2]

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

    logging.info('  Loading UVTable')

    # Get extension removing compressed part
    base, extension = os.path.splitext(data_file)
    if extension in {'.gz', '.bz2'}:
        extension = os.path.splitext(base)[1]
        if extension not in {'.txt', '.dat'}:
            raise ValueError("Compressed UV tables (`.gz` or `.bz2`) must be in "
                             "one of the formats `.txt` or `.dat`.")

    if extension in {'.txt', '.dat'}:
        u, v, re, im, weights = np.genfromtxt(data_file).T
        vis = re + 1j*im

    elif extension == '.npz':
        dat = np.load(data_file)
        u, v, vis, weights = [dat[i] for i in ['u', 'v', 'V', 'weights']]
        if not np.iscomplexobj(vis):
            raise ValueError("You provided a UVTable with the extension {}."
                             " This extension requires the UVTable's variable 'V' to be"
                             " complex (of the form Re(V) + Im(V) * 1j).".format(extension))

    else:
        raise ValueError("You provided a UVTable with the extension {}."
                         " Please provide it as a `.txt`, `.dat`, or `.npz`."
                         " Formats .txt and .dat may optionally be"
                         " compressed (`.gz`, `.bz2`).".format(extension))

    return u, v, vis, weights


def save_uvtable(filename, u, v, vis, weights):
    r"""Save a uvtable to file.

    Parameters
    ----------
    filename : string
        File to save the uvtable to.
    u, v : array, unit = :math:`\lambda`
        u and v coordinates of observations
    vis : array, unit = Jy
        Complex visibilities
    weights : array, unit = Jy^-2
        Weights on the visibilities, of the form
        :math:`1 / \sigma^2`
    """

    extension = os.path.splitext(filename)[1]
    if extension not in {'.txt', '.dat', '.npz'}:
        raise ValueError("file extension must be 'npz', 'txt', or 'dat'.")

    if extension in {'.txt', '.dat'}:
        header = 'u [lambda]\tv [lambda]\tRe(V)  [Jy]\tIm(V) [Jy]\tWeight [Jy^-2]'

        np.savetxt(filename,
                   np.stack([u, v, vis.real, vis.imag,
                             weights], axis=-1),
                   header=header)

    elif extension == '.npz':
        np.savez(filename,
                 u=u, v=v, V=vis, weights=weights,
                 units={'u': 'lambda', 'v': 'lambda',
                        'V': 'Jy', 'weights': "Jy^-2"})


def load_sol(sol_file):
    """Load a frank solution object

    Parameters
    ----------
    sol_file : string
        Filename for frank solution object, '*.obj'

    Returns
    ----------
    sol : _HankelRegressor object
        frank solution object
        (see frank.radial_fitters.FrankFitter)
    """

    sol = np.load(sol_file, allow_pickle=True)

    return sol


def save_fit(u, v, vis, weights, sol, prefix, save_solution=True,
             save_profile_fit=True, save_vis_fit=True, save_uvtables=True,
             save_iteration_diag=False, iteration_diag=None,
             format='npz',
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
        Base part of the filename to which files will be saved
    save_solution : bool
        Whether to save `sol` object (see frank.radial_fitters.FrankFitter)
    save_profile_fit : bool
        Whether to save fitted brightness profile
    save_vis_fit : bool
        Whether to save fitted visibility distribution
        NOTE: This is deprojected
    save_uvtables : bool
        Whether to save fitted and residual UVTables
        NOTE: These are reprojected
    save_iteration_diag : bool
        Whether to save diagnostics of the fit iteration
    iteration_diag : dict
        Diagnostics of the fit iteration
        (see radial_fitters.FrankFitter.fit)
    format : string, default = 'npz'
        File format in which to save the fit's output UVTable(s)
    """

    logging.info('  Saving fit results to {}*'.format(prefix))


    if not format in {'txt', 'dat', 'npz'}:
        raise ValueError("'format' must be 'npz', 'txt', or 'dat'.")

    if save_solution:
        with open(prefix + '_frank_sol.obj', 'wb') as f:
            pickle.dump(sol, f)

    if save_iteration_diag:
        with open(prefix + '_frank_iteration_diagnostics.obj', 'wb') as f:
            pickle.dump(iteration_diag, f)

    if save_profile_fit:
        np.savetxt(prefix + '_frank_profile_fit.txt',
                   np.array([sol.r, sol.I, np.diag(sol.covariance)**.5]).T,
                   header='r [arcsec]\tI [Jy/sr]\tI_uncer [Jy/sr]')

    if save_vis_fit:
        np.savetxt(prefix + '_frank_vis_fit.' + format,
                   np.array([sol.q, sol.predict_deprojected(sol.q).real]).T,
                   header='Baseline [lambda]\tProjected Re(V) [Jy]')


    if save_uvtables:
        logging.info('    Saving fit and residual UVTables. N.B.: These will'
                     ' be of comparable size to your input UVTable')

        V_pred = sol.predict(u, v)

        save_uvtable(prefix + '_frank_uv_fit.' + format,
                     u, v, V_pred, weights)
        save_uvtable(prefix + '_frank_uv_resid.' + format,
                     u, v, vis - V_pred, weights)
