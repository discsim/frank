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
frank_path = os.path.dirname(frank.__file__)


def get_default_parameter_file():
    """Get the path to the default parameter file"""
    return os.path.join(frank_path, 'default_parameters.json')


def load_default_parameters():
    """Load the default parameters"""
    return json.load(open(get_default_parameter_file(), 'r'))


def get_parameter_descriptions():
    """Get the description for paramters"""
    with open(os.path.join(frank_path, 'parameter_descriptions.json')) as f:
        param_descrip = json.load(f)
    return param_descrip


def helper():
    param_descrip = get_parameter_descriptions()

    print("""
         Fit a 1D radial brightness profile with Frankenstein (frank) from the
         terminal with `python -m frank.fit`. A .json parameter file is required;
         the default is default_parameters.json and is
         of the form:\n\n {}""".format(json.dumps(param_descrip, indent=4)))


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
    parser.add_argument("--print_parameter_description", default=None,
                        action="store_true",
                        help="Print the full description of each of the fit "
                        "parameters")

    args = parser.parse_args()

    if args.print_parameter_description:
        helper()
        exit()

    model = json.load(open(args.parameter_filename, 'r'))

    if args.uvtable_filename:
        model['input_output']['uvtable_filename'] = args.uvtable_filename

    if ('uvtable_filename' not in model['input_output'] or
            not model['input_output']['uvtable_filename']):
        raise ValueError("uvtable_filename isn't specified."
                         " Set it in the parameter file or run frank with"
                         " python -m frank.fit -uv <uvtable_filename>")

    uv_path = model['input_output']['uvtable_filename']
    if not model['input_output']['save_dir']:
        # If not specified, use the UVTable directory as the save directory
        model['input_output']['save_dir'] = os.path.dirname(uv_path)

    # Add a save prefix to the json for later use.
    model['input_output']['save_prefix'] = save_prefix =  \
        os.path.join(model['input_output']['save_dir'],
                     os.path.splitext(os.path.basename(uv_path))[0])

    log_path = save_prefix + '_frank_fit.log'
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(log_path, mode='w'),
                            logging.StreamHandler()]
                        )

    logging.info('\nRunning frank on'
                 ' {}'.format(model['input_output']['uvtable_filename']))

    # Sanity check some of json parameters
    if model['plotting']['diag_plot']:
        plotting = model['plotting']

        if plotting['iter_plot_range'] is not None:
            err = ValueError("iter_plot_range should be 'null' (None) "
                             "or a list specifying the start and end "
                             "points of the range to be plotted".)
            try:
                if len(plotting['iter_plot_range']) != 2:
                    raise err
            except TypeError:
                raise err

    param_path = save_prefix + '_frank_used_pars.json'
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
    geometry_type: string, from {'known', 'gaussian'}
        Specifies how the geometry is determined. Options:
            'known' : The user-provided geometry will be used
            'gaussian' : Determine the geometry by fitting a Gaussian
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
        logging.info('    Using your provided geometry for deprojection')
        if all(x == 0 for x in (inc, pa, dra, ddec)):
            logging.info("      N.B.: All geometry parameters are 0, so I won't"
                         " apply any geometry correction to the visibilities")
        geom = geometry.FixedGeometry(inc, pa, dra, ddec)

    elif geometry_type == 'gaussian':
        if fit_phase_offset:
            logging.info('    Fitting Gaussian to determine geometry')
            geom = geometry.FitGeometryGaussian()

        else:
            logging.info('    Fitting Gaussian to determine geometry'
                         ' (not fitting for phase center)')
            geom = geometry.FitGeometryGaussian(phase_centre=(dra, ddec))

        t1 = time.time()
        geom.fit(u, v, vis, weights)
        logging.info('    Time taken for geometry %.1f sec' %
                     (time.time() - t1))
    else:
        raise ValueError("geometry_type must be one of 'known' or 'gaussian'")

    logging.info('    Using: inc  = {:.2f} deg,\n           PA   = {:.2f} deg,\n'
                 '           dRA  = {:.2e} mas,\n'
                 '           dDec = {:.2e} mas'.format(geom.inc, geom.PA,
                                                       geom.dRA*1e3,
                                                       geom.dDec*1e3))

    # Store geometry
    geom = geom.clone()

    return geom


def perform_fit(u, v, vis, weights, geom, rout, n, alpha, wsmooth, max_iter,
                return_iteration_diag, diag_plot
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
    max_iter : int
        Maximum number of fit iterations
    return_iteration_diag : bool
        Whether to return diagnostics of the fit iteration
        (see radial_fitters.FrankFitter.fit)
    diag_plot : bool
        A check for whether to return diagnostics of the fit iteration
        (if frank.make_figs.make_diag_fig is being called,
        return_iteration_diag must be True)

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

    need_iterations = return_iteration_diag or diag_plot

    FF = radial_fitters.FrankFitter(Rmax=rout, N=n, geometry=geom,
                                    alpha=alpha, weights_smooth=wsmooth,
                                    max_iter=max_iter,
                                    store_iteration_diagnostics=need_iterations
                                    )

    t1 = time.time()
    sol = FF.fit(u, v, vis, weights)
    logging.info('    Time taken to fit profile (with {:.0e} visibilities and'
                 '{:d} collocation points) {:.1f} sec'.format(len(vis), n,
                                                              time.time() - t1))

    if need_iterations:
        return sol, FF.iteration_diagnostics
    else:
        return [sol, ]


def output_results(u, v, vis, weights, sol, iteration_diag, iter_plot_range,
                   bin_widths, save_prefix,
                   save_profile_fit, save_vis_fit, save_uvtables,
                   save_iteration_diag, full_plot, quick_plot, diag_plot,
                   force_style=True, dist=None
                   ):
    r"""
    Save datafiles of fit results; generate and save figures of fit results (see
    frank.io.save_fit, frank.make_figs.make_full_fig,
    frank.make_figs.make_quick_fig, frank.make_figs.make_diag_fig)

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
    iteration_diag : _HankelRegressor object
        Diagnostics of the fit iteration
        (see radial_fitters.FrankFitter.fit)
    iter_plot_range : list or None
        Range of iterations in the fit over which to
        plot brightness profile and power spectrum reconstructions. If None,
        then the full range will be plotted.
    bin_widths : list, unit = \lambda
        Bin widths in which to bin the observed visibilities
    save_prefix : string
        Prefix for output filenames
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
    diag_plot : bool
        Whether to make a figure showing convergence diagnostics for the fit
    force_style: bool, default = True
        Whether to use preconfigured matplotlib rcParams in generated figures
    dist : float, optional, unit = AU, default = None
        Distance to source, used to show second x-axis for brightness profile
    """

    logging.info('  Plotting results')

    figs = []
    axes = []

    if quick_plot:
        logging.info('    Making quick figure')
        quick_fig, quick_axes = make_figs.make_quick_fig(u, v, vis, weights, sol, bin_widths, dist,
                                                         force_style, save_prefix
                                                         )

        figs.append(quick_fig)
        axes.append(quick_axes)

    if full_plot:
        logging.info('    Making full figure')
        full_fig, full_axes = make_figs.make_full_fig(u, v, vis, weights, sol, bin_widths, dist,
                                                      force_style, save_prefix
                                                      )

        figs.append(full_fig)
        axes.append(full_axes)

    if diag_plot:
        if iter_plot_range is not None:
            if iter_plot_range[1] > iteration_diag['num_iterations']:
                if iter_plot_range[0] < iteration_diag['num_iterations']:
                    logging.info('    Upper limit of iteration plot range '
                                 'exceeds number of iterations, truncating '
                                 'to the number of iterations used')
                    iter_plot_range = [iter_plot_range[0],
                                       iteration_diag['num_iterations']]
                else:
                    logging.info('    Lower limit of iteration plot range '
                                 'exceeds number of iterations, no iterations '
                                 'will be plotted')
                    iter_plot_range = [iteration_diag['num_iterations'],
                                       iteration_diag['num_iterations']]

        diag_fig, diag_axes = make_figs.make_diag_fig(sol.r, sol.q,
                                                      iteration_diag,
                                                      iter_plot_range,
                                                      force_style, save_prefix
                                                      )

        figs.append(diag_fig)
        axes.append(diag_axes)

    logging.info('  Saving results')

    io.save_fit(u, v, vis, weights, sol, save_prefix,
                save_profile_fit, save_vis_fit, save_uvtables,
                save_iteration_diag, iteration_diag
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
                                             model['hyperpriors']['max_iter'],
                                             model['input_output']['iteration_diag'],
                                             model['plotting']['diag_plot']
                                             )

    figs = output_results(u, v, vis, weights, sol, iteration_diagnostics,
                          model['plotting']['iter_plot_range'],
                          model['plotting']['bin_widths'],
                          model['input_output']['save_prefix'],
                          model['input_output']['save_vis_fit'],
                          model['input_output']['save_uvtables'],
                          model['input_output']['iteration_diag'],
                          model['plotting']['full_plot'],
                          model['plotting']['quick_plot'],
                          model['plotting']['diag_plot'],
                          model['plotting']['force_style'],
                          model['plotting']['dist']
                          )

    logging.info("IT'S ALIVE!!\n")

    return figs


if __name__ == "__main__":
    main()
