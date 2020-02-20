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
"""This module runs Frankenstein to fit a source's 1D radial brightness profile.
   A default parameter file is used that specifies all options to run the fit
   and output results. Alternatively a custom parameter file can be provided.
"""

from frank import io, make_figs
import os
import sys
import time
import json
import numpy as np

import logging

import frank
frank_path = os.path.dirname(frank.__file__)  # TODO


def helper():
    with open(frank_path + '/parameter_descriptions.json') as f:
        param_descrip = json.load(f)

    print("""
         Fit a 1D radial brightness profile with Frankenstein (frank) from the
         terminal with `python -m frank.fit`. A .json parameter file is required;
         the default is default_parameters.json and is of the form:\n\n""",
          json.dumps(param_descrip, indent=4))  # TODO


def parse_parameters():
    """
    Read in a .json parameter file to set the fit parameters

    Parameters
    ----------
    parameter_filename : string, default `default_parameters.json`
            Parameter file (.json; see frank.fit.helper)
    uvtable_filename : string
            UVTable file with data to be fit (.txt, .dat, .npy, or .npz).
            The UVTable column format should be:
            u [lambda] v [lambda] Re(V) [Jy] Im(V) [Jy] Weight [Jy^-2]

    Returns
    -------
    model : dict
            Dictionary containing model parameters the fit uses
    """

    import argparse

    default_param_file = os.path.join(frank_path, 'default_parameters.json')

    parser = argparse.ArgumentParser("Run a Frank fit, by default using"
                                     " parameters in default_parameters.json")
    parser.add_argument("-p", "--parameter_filename",
                        default=default_param_file, type=str,
                        help="Parameter file (.json; see frank.fit.helper)")
    parser.add_argument("-uv", "--uvtable_filename", default=None, type=str,
                        help="UVTable file with data to be fit. See"
                             " frank.io.load_uvtable")

    args = parser.parse_args()
    model = json.load(open(args.parameter_filename, 'r'))

    if args.uvtable_filename:
        model['input_output']['uvtable_filename'] = args.uvtable_filename

    if ('uvtable_filename' not in model['input_output'] or
            not model['input_output']['uvtable_filename']):
        raise ValueError("    uvtable_filename isn't specified."
                         " Set it in the parameter file or run frank with"
                         " python -m frank.fit -uv <uvtable_filename>")

    if not model['input_output']['save_dir']:
        # Use the uv table location as save point:
        uv_path = model['input_output']['uvtable_filename']
        if uv_path:
            model['input_output']['save_dir'] = os.path.dirname(uv_path)

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(
                                os.path.join(model['input_output']['save_dir'],
                                             'frank_fit.log'),
                                mode='w'),
                            logging.StreamHandler()]
                        )

    logging.info('\nRunning frank on %s'
                 % model['input_output']['uvtable_filename'])

    param_path = os.path.join(model['input_output']['save_dir'],
                              'frank_used_pars.json')
    logging.info(
        '  Saving parameters to be used in fit to {}'.format(param_path))
    with open(param_path, 'w') as f:
        json.dump(model, f, indent=4)

    return model


def load_data(data_file):
    r"""
    Read in a UVTable with data to be fit. See frank.io.load_uvtable

    Parameters
    ----------
    data_file : string
          UVTable with columns:
          u [lambda]  v [lambda]  Re(V) [Jy]  Im(V) [Jy] Weight [Jy^-2]

    Returns
    -------
    u, v : array, unit = :math:`\lambda`
          u and v coordinates of observations
    vis : array, unit = Jy
          Observed visibilities (complex: real + imag * 1j)
    weights : array, unit = Jy^-2
          Weights assigned to observed visibilities, of the form
          :math:`1 / \sigma^2`
    """
    logging.info('  Loading UVTable')
    u, v, vis, weights = io.load_uvtable(data_file)

    return u, v, vis, weights


def determine_geometry(u, v, vis, weights, inc, pa, dra, ddec, geometry_type,
                       fit_phase_offset
                       ):
    r"""
    Determine the source geometry (inclination, position angle, phase offset)

    Parameters
    ----------
    u, v : array, unit = :math:`\lambda`
        u and v coordinates of observations
    vis : array, unit = Jy
        Observed visibilities (complex: real + imag * 1j)
    weights : array, unit = Jy^-2
        Weights assigned to observed visibilities, of the form
        :math:`1 / \sigma^2`
    inc: float, unit = deg
        Source inclination
    pa : float, unit = deg
        Source position angle
    dra : float, unit = arcsec
        Source right ascension offset from 0
    ddec : float, unit = arcsec
        Source declination offset from 0
    geometry_type: string, from {'known', 'gaussian' }
        Specifies how the geometry is determined. Options:
            'known' :  a prescribed geometry
            'gaussian' : determine geometry by fitting a Gaussian
    fit_phase_offset: bool
        Whether to fit for the source's right ascension offset and declination
        offset from 0

    Returns
    -------
    geom : SourceGeometry object
          Fitted geometry (see frank.geometry.SourceGeometry)
    """

    from frank import geometry

    logging.info('  Determining disc geometry')

    if geometry_type == 'known':
        geom = geometry.FixedGeometry(inc, pa, dra, ddec)

    elif geometry_type == 'gaussian':
        if fit_phase_offset:
            geom = geometry.FitGeometryGaussian()

        else:
            geom = geometry.FitGeometryGaussian(
                phase_centre=(dra, ddec))

        t1 = time.time()
        geom.fit(u, v, vis, weights)
        logging.info('    Time taken to fit geometry %.1f sec' % (time.time()
                                                                  - t1))
    else:
        raise ValueError("geometry_type must be one of 'known' or 'gaussian'")

    logging.info('    Using: inc  = %.2f deg,\n           PA   = %.2f deg,\n'
                 '           dRA  = %.2e mas,\n           dDec = %.2e mas'
                 % (geom.inc, geom.PA, geom.dRA*1e3, geom.dDec*1e3))

    geom = geom.clone()
    logging.info('    Storing disc geometry to use for fit')

    return geom


def perform_fit(u, v, vis, weights, geom, rout, n, alpha, wsmooth,
                iteration_diag=False
                ):
    r"""
    Deproject the observed visibilities and fit them for the brightness profile

    Parameters
    ----------
    u, v : array, unit = :math:`\lambda`
          u and v coordinates of observations
    vis : array, unit = Jy
          Observed visibilities (complex: real + imag * 1j)
    weights : array, unit = Jy^-2
          Weights assigned to observed visibilities, of the form
          :math:`1 / \sigma^2`
    geom : SourceGeometry object
          Fitted geometry (see frank.geometry.SourceGeometry)
    rout : float, unit = arcsec
          Maximum disc radius in the fit (best to overestimate size of source)
    n : int
          Number of collocation points used in the fit
          (suggested range 100 - 300)
    alpha : float
          Order parameter for the power spectrum's inverse Gamma prior
          (suggested range 1.00 - 1.50)
    wsmooth : float
          Strength of smoothing applied to the power spectrum
          (suggested range 10^-4 - 10^-1)
    iteration_diag : bool, default = False
          Whether to return diagnostics of the fit iteration
          (see radial_fitters.FrankFitter.fit)

    Returns
    -------
    sol : _HankelRegressor object
          Reconstructed profile using Maximum a posteriori power spectrum
          (see frank.radial_fitters.FrankFitter)
    iteration_diag : _HankelRegressor object
          Diagnostics of the fit iteration
          (see radial_fitters.FrankFitter.fit)
    """

    from frank import radial_fitters

    logging.info('  Fitting for brightness profile')

    FF = radial_fitters.FrankFitter(Rmax=rout, N=n, geometry=geom,
                                    alpha=alpha, weights_smooth=wsmooth
                                    )

    t1 = time.time()
    sol = FF.fit(u, v, vis, weights)
    logging.info('    Time taken to fit profile (with %.0e visibilities and %s'
                 ' collocation points) %.1f sec' % (len(vis), n, time.time() - t1))

    if iteration_diag:
        return sol, FF.iteration_diagnostics
    else:
        return sol, None


def output_results(u, v, vis, weights, geom, sol, bin_widths,
                   save_dir, uvtable_filename, save_profile_fit, save_vis_fit,
                   save_uvtables, save_iteration_diag, full_plot, quick_plot,
                   force_style=True, dist=None
                   ):
    r"""
    Save datafiles of fit results; generate and save figures of fit results (see
    frank.io.save_fit, frank.make_figs.make_full_fig and
    frank.make_figs.make_quick_fig)

    Parameters
    ----------
    u, v : array, unit = :math:`\lambda`
          u and v coordinates of observations
    vis : array, unit = Jy
          Observed visibilities (complex: real + imag * 1j)
    weights : array, unit = Jy^-2
          Weights assigned to observed visibilities, of the form
          :math:`1 / \sigma^2`
    geom : SourceGeometry object
          Fitted geometry (see frank.geometry.SourceGeometry)
    sol : _HankelRegressor object
          Reconstructed profile using Maximum a posteriori power spectrum
          (see frank.radial_fitters.FrankFitter)
    bin_widths : list, unit = \lambda
          Bin widths in which to bin the observed visibilities
    save_dir : string
          Directory in which output datafiles and figures are saved
    uvtable_filename : string
          Filename for observed UVTable. The saved datafiles and figures use
          this as their filename prefix
    save_profile_fit : bool
          Whether to save fitted brightness profile
    save_vis_fit : bool
          Whether to save fitted visibility distribution
    save_uvtables : bool
          Whether to save fitted and residual UV tables.
          NOTE: These are reprojected
    save_iteration_diag : bool
          Whether to save diagnostics of the fit iteration
    full_plot : bool
          Whether to make a figure more fully showing the fit and its
          diagnostics
    quick_plot : bool
          Whether to make a figure showing the simplest plots of the fit
    force_style: bool, default = True
          Whether to use preconfigured matplotlib rcParams in generated figures
    dist : float, optional, unit = AU, default = None
          Distance to source, used to show second x-axis for brightness profile
    """
    logging.info('  Plotting results')

    figs = []
    axes = []

    save_prefix = \
        os.path.join(save_dir,
                     os.path.splitext(os.path.basename(uvtable_filename))[0])

    if quick_plot:
        quick_fig, quick_axes = \
            make_figs.make_quick_fig(u, v, vis, weights,
                                     sol, bin_widths, dist, force_style, save_prefix
                                     )

        figs.append(quick_fig)
        axes.append(quick_axes)

    if full_plot:
        full_fig, full_axes = \
            make_figs.make_full_fig(u, v, vis, weights,
                                    sol, bin_widths, dist, force_style, save_prefix
                                    )

        figs.append(full_fig)
        axes.append(axes)

    logging.info('  Saving results')
    io.save_fit(u, v, vis, weights, sol, save_prefix,
                save_profile_fit, save_vis_fit, save_uvtables,
                save_iteration_diag
                )

    return figs, axes


def main():
    model = parse_parameters()

    u, v, vis, weights = load_data(model['input_output']['uvtable_filename'])

    geom = determine_geometry(u, v, vis, weights,
                              model['geometry']['inc'],
                              model['geometry']['pa'],
                              model['geometry']['dra'],
                              model['geometry']['ddec'],
                              model['geometry']['geometry_type'],
                              model['geometry']['fit_phase_offset']
                              )

    sol, iteration_diagnostics = perform_fit(u, v, vis, weights, geom,
                                             model['hyperpriors']['rout'],
                                             model['hyperpriors']['n'],
                                             model['hyperpriors']['alpha'],
                                             model['hyperpriors']['wsmooth'],
                                             model['input_output']['iteration_diag']
                                             )

    figs = output_results(u, v, vis, weights, geom, sol,
                          model['plotting']['bin_widths'],
                          model['input_output']['save_dir'],
                          model['input_output']['uvtable_filename'],
                          model['input_output']['save_profile_fit'],
                          model['input_output']['save_vis_fit'],
                          model['input_output']['save_uvtables'],
                          model['input_output']['iteration_diag'],
                          model['plotting']['full_plot'],
                          model['plotting']['quick_plot'],
                          model['plotting']['force_style'],
                          model['plotting']['dist']
                          )

    logging.info("IT'S ALIVE!!\n")


if __name__ == "__main__":
    main()
