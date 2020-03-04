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

import os
import sys
import time
import json
import numpy as np

import logging

import frank
frank_path = os.path.dirname(frank.__file__)

from frank import io, geometry, make_figs, radial_fitters, utilities


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


def parse_parameters(*args):
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

    parser = argparse.ArgumentParser("Run a Frankenstein fit, by default using"
                                     " parameters in default_parameters.json")
    parser.add_argument("-p", "--parameter_filename",
                        default=default_param_file, type=str,
                        help="Parameter file (.json; see frank.fit.helper)")
    parser.add_argument("-uv", "--uvtable_filename", default=None, type=str,
                        help="UVTable file with data to be fit. See"
                             " frank.io.load_uvtable")
    parser.add_argument("-desc", "--print_parameter_description", default=None,
                        action="store_true",
                        help="Print the full description of all fit parameters")

    args = parser.parse_args(*args)

    if args.print_parameter_description:
        helper()
        exit()

    model = json.load(open(args.parameter_filename, 'r'))

    if args.uvtable_filename:
        model['input_output']['uvtable_filename'] = args.uvtable_filename

    if ('uvtable_filename' not in model['input_output'] or
            not model['input_output']['uvtable_filename']):
        raise ValueError("uvtable_filename isn't specified."
                         " Set it in the parameter file or run Frankenstein with"
                         " python -m frank.fit -uv <uvtable_filename>")

    uv_path = model['input_output']['uvtable_filename']
    if not model['input_output']['save_dir']:
        # If not specified, use the UVTable directory as the save directory
        model['input_output']['save_dir'] = os.path.dirname(uv_path)

    # Add a save prefix to the .json parameter file for later use
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

    logging.info('\nRunning Frankenstein on'
                 ' {}'.format(model['input_output']['uvtable_filename']))

    # Sanity check some of the .json parameters
    if model['plotting']['diag_plot']:
        if model['plotting']['iter_plot_range'] is not None:
            err = ValueError("iter_plot_range should be 'null' (None)"
                             " or a list specifying the start and end"
                             " points of the range to be plotted")
            try:
                if len(model['plotting']['iter_plot_range']) != 2:
                    raise err
            except TypeError:
                raise err

    if model['modify_data']['cut_data']:
        if model['modify_data']['cut_range'] is not None:
            err = ValueError("cut_range should be 'null' (None)"
                             " or a list specifying the low and high"
                             " baselines [unit: \\lambda] outside of which the"
                             " data will be truncated before fitting")
            try:
                if len(model['modify_data']['cut_range']) != 2:
                    raise err
            except TypeError:
                raise err

    if model['input_output']['format'] is None:
        model['input_output']['format'] = os.path.splitext(uv_path)[1][1:]

    param_path = save_prefix + '_frank_used_pars.json'

    logging.info(
        '  Saving parameters to be used in fit to {}'.format(param_path))
    with open(param_path, 'w') as f:
        json.dump(model, f, indent=4)

    return model


def load_data(model):
    r"""
    Read in a UVTable with data to be fit. See frank.io.load_uvtable

    Parameters
    ----------
    model : dict
        Dictionary containing model parameters the fit uses

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

    u, v, vis, weights = io.load_uvtable(model['input_output']['uvtable_filename'])

    if model['modify_data']['norm_by_wle']:
        logging.info('  Normalizing u and v by observing wavelength of'
                     ' {} m'.format(model['modify_data']['wle']))

        u /= model['modify_data']['wle'] # TODO: should go in function in io
        v /= model['modify_data']['wle']

    if model['modify_data']['cut_data']:
        logging.info('  Cutting data outside of the minmum and maximum baselines'
                     ' of {} and {}'
                     ' klambda'.format(model['modify_data']['cut_range'][0] / 1e3,
                                      model['modify_data']['cut_range'][1] / 1e3))

        baselines = np.hypot(u, v)
        above_lo = baselines >= model['modify_data']['cut_range'][0]
        below_hi = baselines <= model['modify_data']['cut_range'][1]
        in_range = above_lo & below_hi
        u, v, vis, weights = [x[in_range] for x in [u, v, vis, weights]]

    return u, v, vis, weights


def apply_correction_to_weights(u, v, ReV, weights, nbins=300): # TODO: should call func in utilities.py
    r"""
    Estimate and apply a correction factor to the data's weights by comparing
    binnings of the real component of the visibilities under different
    weightings. This is useful for mock datasets in which the weights are all
    unity.

    Parameters
    ----------
    u, v : array, unit = :math:`\lambda`
        u and v coordinates of observations

    ReV : array, unit = Jy
        Real component of observed visibilities

    weights : array, unit = Jy^-2
        Weights assigned to observed visibilities, of the form
        :math:`1 / \sigma^2`

    nbins : int, default=300
        Number of bins used to construct the histograms

    Returns
    -------
    wcorr_estimate : float
        Correction factor by which to adjust the weights

    weights_corrected : array, unit = Jy^-2
        Corrected weights assigned to observed visibilities, of the form
        :math:`1 / \sigma^2`
    """

    logging.info('  Estimating, applying correction factor to the weights') # TODO: should go in utilities.py

    baselines = np.hypot(u, v)
    mu, edges = np.histogram(np.log10(baselines), weights=ReV, bins=nbins)
    mu2, edges = np.histogram(np.log10(baselines), weights=ReV ** 2, bins=nbins)
    N, edges = np.histogram(np.log10(baselines), bins=nbins)

    centres = 0.5 * (edges[1:] + edges[:-1])

    mu /= np.maximum(N, 1)
    mu2 /= np.maximum(N, 1)

    sigma = (mu2 - mu ** 2) ** 0.5
    wcorr_estimate = sigma[np.where(sigma > 0)].mean()

    weights_corrected = weights / wcorr_estimate ** 2

    return wcorr_estimate, weights_corrected


def determine_geometry(u, v, vis, weights, model):
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
    model : dict
        Dictionary containing model parameters the fit uses

    Returns
    -------
    geom : SourceGeometry object
        Fitted geometry (see frank.geometry.SourceGeometry)
    """

    logging.info('  Determining disc geometry') # TODO: should go in geometry.py

    if model['geometry']['type'] == 'known':
        logging.info('    Using your provided geometry for deprojection')
        if all(x == 0 for x in (model['geometry']['inc'],
                                model['geometry']['pa'],
                                model['geometry']['dra'],
                                model['geometry']['ddec'])
                                ):
            logging.info("      N.B.: All geometry parameters are 0 -->"
                         " No geometry correction will be applied to the"
                         " visibilities"
                         )

        geom = geometry.FixedGeometry(model['geometry']['inc'],
                                      model['geometry']['pa'],
                                      model['geometry']['dra'],
                                      model['geometry']['ddec']
                                      )

    elif model['geometry']['type'] == 'gaussian':
        if model['geometry']['fit_phase_offset']:
            logging.info('    Fitting Gaussian to determine geometry') # TODO: should go in geometry.py
            geom = geometry.FitGeometryGaussian()

        else:
            logging.info('    Fitting Gaussian to determine geometry'
                         ' (not fitting for phase center)') # TODO: should go in geometry.py
            geom = geometry.FitGeometryGaussian(phase_centre=(model['geometry']['dra'],
                                                              model['geometry']['ddec']))

        t1 = time.time()
        geom.fit(u, v, vis, weights)
        logging.info('    Time taken for geometry %.1f sec' %
                     (time.time() - t1)) # TODO: should go in geometry.py

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


def perform_fit(u, v, vis, weights, geom, model):
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
    model : dict
        Dictionary containing model parameters the fit uses

    Returns
    -------
    sol : _HankelRegressor object
        Reconstructed profile using Maximum a posteriori power spectrum
        (see frank.radial_fitters.FrankFitter)
    iteration_diag : _HankelRegressor object
        Diagnostics of the fit iteration
        (see radial_fitters.FrankFitter.fit)
    """

    logging.info('  Fitting for brightness profile') # TODO: should go in frankfitter

    need_iterations = model['input_output']['iteration_diag'] or \
                      model['plotting']['diag_plot']

    FF = radial_fitters.FrankFitter(Rmax=model['hyperpriors']['rout'],
                                    N=model['hyperpriors']['n'],
                                    geometry=geom,
                                    alpha=model['hyperpriors']['alpha'],
                                    weights_smooth=model['hyperpriors']['wsmooth'],
                                    tol=model['hyperpriors']['iter_tol'],
                                    max_iter=model['hyperpriors']['max_iter'],
                                    store_iteration_diagnostics=need_iterations
                                    )

    t1 = time.time()
    sol = FF.fit(u, v, vis, weights)
    logging.info('    Time taken to fit profile (with {:.0e} visibilities and'
                 ' {:d} collocation points) {:.1f} sec'.format(len(vis),
                                                               model['hyperpriors']['n'],
                                                               time.time() - t1)
                                                               ) # TODO: should go in frankfitter

    if need_iterations:
        return sol, FF.iteration_diagnostics
    else:
        return [sol, None]


def output_results(u, v, vis, weights, sol, iteration_diag, model):
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
    model : dict
        Dictionary containing model parameters the fit uses

    Returns
    -------
    figs : Matplotlib `.Figure` instance
        All produced figures, including the GridSpecs
    axes : Matplotlib `~.axes.Axes` class
        Axes for each of the produced figures
    """

    logging.info('  Plotting results') # TODO: should go in plot.py

    figs, axes = [], []

    if model['plotting']['quick_plot']:
        logging.info('    Making quick figure') # TODO: should go in make_quick_fig
        quick_fig, quick_axes = make_figs.make_quick_fig(u, v, vis, weights, sol,
                                                         model['plotting']['bin_widths'],
                                                         model['plotting']['dist'],
                                                         model['plotting']['force_style'],
                                                         model['input_output']['save_prefix']
                                                         )

        figs.append(quick_fig)
        axes.append(quick_axes)

    if model['plotting']['full_plot']:
        logging.info('    Making full figure') # TODO: should go in make_full_fig
        full_fig, full_axes = make_figs.make_full_fig(u, v, vis, weights, sol,
                                                      model['plotting']['bin_widths'],
                                                      model['plotting']['dist'],
                                                      model['plotting']['force_style'],
                                                      model['input_output']['save_prefix']
                                                      )

        figs.append(full_fig)
        axes.append(full_axes)

    if model['plotting']['diag_plot']:
        logging.info('    Making diagnostic figure') # TODO: should go in make_full_fig
        if model['plotting']['iter_plot_range'] is None: # TODO: should go in make_diag_plot
            logging.info("      diag_plot is 'true' in your parameter file but"
                         " iter_plot_range is 'null' --> Defaulting to"
                         " plotting all iterations")

            model['plotting']['iter_plot_range'] = [0, iteration_diag['num_iterations']]

        else:
            if model['plotting']['iter_plot_range'][0] > iteration_diag['num_iterations']:
                logging.info('      iter_plot_range[0] in your parameter file'
                             ' exceeds the number of fit iterations -->'
                             ' Defaulting to plotting all iterations')

                model['plotting']['iter_plot_range'] = [0, iteration_diag['num_iterations']]

        diag_fig, diag_axes = make_figs.make_diag_fig(sol.r, sol.q,
                                                      iteration_diag,
                                                      model['plotting']['iter_plot_range'],
                                                      model['plotting']['force_style'],
                                                      model['input_output']['save_prefix']
                                                      )

        figs.append(diag_fig)
        axes.append(diag_axes)

    logging.info('  Saving results') # TODO: should go in io func

    io.save_fit(u, v, vis, weights, sol,
                model['input_output']['save_prefix'],
                model['input_output']['save_solution'],
                model['input_output']['save_profile_fit'],
                model['input_output']['save_vis_fit'],
                model['input_output']['save_uvtables'],
                model['input_output']['iteration_diag'],
                iteration_diag,
                model['input_output']['format']
                )

    return figs, axes


def perform_bootstrap(u, v, vis, weights, geom, model):
    r"""
    Perform a bootstrap analysis for the Franktenstein fit to a dataset

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
    model : dict
        Dictionary containing model parameters the fit uses

    Returns
    -------
    boot_fig : Matplotlib `.Figure` instance
        The produced figure, including the GridSpec
    boot_axes : Matplotlib `~.axes.Axes` class
        The axes of the produced figure
    """
    profiles_bootstrap = []

    for ii in range(model['analysis']['n_trials']):
        logging.info(' Bootstrap trial {} of {}'.format(ii + 1,
                                                 model['analysis']['n_trials']))

        utilities.draw_bootstrap_sample(u, v, vis, weights)

        sol, iteration_diagnostics = perform_fit(u, v, vis, weights, geom, model)
        profiles_bootstrap.append(sol.mean)

    profiles_path = model['input_output']['save_prefix'] + \
                        '_bootstrap_profiles.txt'
    collocation_points_path = model['input_output']['save_prefix'] + \
                                  '_bootstrap_collocation_pts.txt'

    logging.info(' Bootstrap complete. Saving fitted brightness profiles'
                 ' and the common set of collocation points')

    np.savetxt(profiles_path, profiles_bootstrap)
    np.savetxt(collocation_points_path, sol.r)

    logging.info(' Making bootstrap summary figure') # TODO: should go in make_full_fig
    boot_fig, boot_axes = make_figs.make_bootstrap_fig(sol.r,
                                                        profiles_bootstrap,
                                                        model['plotting']['dist'],
                                                        model['plotting']['force_style'],
                                                        model['input_output']['save_prefix']
                                                        )

    return boot_fig, boot_axes


def main(*args):
    """Run the full Frankenstein pipeline to fit a dataset

    Parameters
    ----------
    *args : strings
        Simulates the command line arguments
    """

    model = parse_parameters(*args)

    u, v, vis, weights = load_data(model)

    if model['modify_data']['correct_weights']:
        wcorr_estimate, weights = apply_correction_to_weights(u, v, vis.real,
                                                              weights
                                                              )

    geom = determine_geometry(u, v, vis, weights, model)

    if model['analysis']['bootstrap']:
        boot_fig, boot_axes = perform_bootstrap(u, v, vis, weights, geom, model)
        return boot_fig, boot_axes

    else:
        sol, iteration_diagnostics = perform_fit(u, v, vis, weights, geom, model)

        figs, axes = output_results(u, v, vis, weights, sol,
                                    iteration_diagnostics, model
                                    )

    logging.info("IT'S ALIVE!!\n")

    return figs, axes


if __name__ == "__main__":
    main()
