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

# Force frank to run on a single thread if we are using it as a library
def _check_and_warn_if_parallel():
    """Check numpy is running in parallel"""
    num_threads = int(os.environ.get('OMP_NUM_THREADS', '1'))
    if num_threads > 1:
        logging.warning("WARNING: You are running frank with "
                        "OMP_NUM_THREADS={}.".format(num_threads) +
                        "The code will likely run faster on a single thread.\n"
                        "Use 'unset OMP_NUM_THREADS' or "
                        "'export OMP_NUM_THREADS=1' to disable this warning.")

import numpy as np

import logging

import frank
from frank import io, geometry, make_figs, radial_fitters, utilities

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
    frank.enable_logging(log_path)

    # Check whether the code runs in parallel now that the logging has been
    # initialized.
    _check_and_warn_if_parallel()


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

    if model['modify_data']['baseline_range'] is not None:
        err = ValueError("baseline_range should be 'null' (None)"
                         " or a list specifying the low and high"
                         " baselines [unit: \\lambda] outside of which the"
                         " data will be truncated before fitting")
        try:
            if len(model['modify_data']['baseline_range']) != 2:
                raise err
        except TypeError:
            raise err

    if model['input_output']['format'] is None:
        path, format = os.path.splitext(uv_path)
        if format in {'.gz', '.bz2'}:
            format = os.path.splitext(path)[1]
        model['input_output']['format'] = format[1:]

    param_path = save_prefix + '_frank_used_pars.json'

    logging.info(
        '  Saving parameters used to {}'.format(param_path))
    with open(param_path, 'w') as f:
        json.dump(model, f, indent=4)

    return model, param_path


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
    ncols : int
          Number of columns in the UVTable
    """

    u, v, vis, weights, ncols = io.load_uvtable(
        model['input_output']['uvtable_filename'])

    return u, v, vis, weights, ncols


def alter_data(u, v, vis, weights, geom, model):
    r"""
    Apply one or more modifications to the data as specified in the parameter file

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
        Fitted geometry (see frank.geometry.SourceGeometry).
    model : dict
        Dictionary containing model parameters the fit uses

    Returns
    -------
    u, v, vis, weights : Parameters as above, with any or all altered according
    to the modification operations specified in model
    """

    if model['modify_data']['normalization_wle'] is not None:
        u, v = utilities.normalize_uv(
            u, v, model['modify_data']['normalization_wle'])

    if model['modify_data']['baseline_range']:
        u, v, vis, weights = \
            utilities.cut_data_by_baseline(u, v, vis, weights,
                                           model['modify_data']['baseline_range'],
                                           geom)

    if model['modify_data']['correct_weights']:
        up, vp = geom.deproject(u,v)
        weights = utilities.estimate_weights(up, vp, vis, use_median=True)

    return u, v, vis, weights


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

    logging.info('  Determining disc geometry')

    if model['geometry']['type'] == 'known':
        logging.info('    Using your provided geometry for deprojection')

        if all(x == 0 for x in (model['geometry']['inc'],
                                model['geometry']['pa'],
                                model['geometry']['dra'],
                                model['geometry']['ddec'])):
            logging.info("      N.B.: All geometry parameters are 0 --> No geometry"
                         " correction will be applied to the visibilities"
                         )

        geom = geometry.FixedGeometry(model['geometry']['inc'],
                                      model['geometry']['pa'],
                                      model['geometry']['dra'],
                                      model['geometry']['ddec']
                                      )

    elif model['geometry']['type'] in ('gaussian', 'nonparametric'):
        t1 = time.time()

        if model['geometry']['initial_guess']:
            guess = [model['geometry']['inc'], model['geometry']['pa'],
                     model['geometry']['dra'], model['geometry']['ddec']]
        else:
            guess = None

        if model['geometry']['fit_phase_offset']:
            phase_centre = (model['geometry']['dra'],
                            model['geometry']['ddec'])
        else:
            phase_centre = None


        if model['geometry']['type'] == 'gaussian':
            geom = geometry.FitGeometryGaussian(
                phase_centre=phase_centre, guess=guess,
            )
        else:
            geom = geometry.FitGeometryFourierBessel(
                model['hyperparameters']['rout'], N=20,
                phase_centre=phase_centre, guess=guess
            )

        geom.fit(u, v, vis, weights)

        logging.info('    Time taken for geometry %.1f sec' %
                     (time.time() - t1))


    else:
        raise ValueError("`geometry : type` in your parameter file must be one of"
                         " 'known', 'gaussian' or 'nonparametric'.")

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
    iteration_diagnostics : _HankelRegressor object
        Diagnostics of the fit iteration
        (see radial_fitters.FrankFitter.fit)
    """

    need_iterations = model['input_output']['iteration_diag'] or \
        model['plotting']['diag_plot']

    t1 = time.time()
    FF = radial_fitters.FrankFitter(Rmax=model['hyperparameters']['rout'],
                                    N=model['hyperparameters']['n'],
                                    geometry=geom,
                                    alpha=model['hyperparameters']['alpha'],
                                    weights_smooth=model['hyperparameters']['wsmooth'],
                                    tol=model['hyperparameters']['iter_tol'],
                                    max_iter=model['hyperparameters']['max_iter'],
                                    store_iteration_diagnostics=need_iterations
                                    )

    sol = FF.fit(u, v, vis, weights)

    if model['hyperparameters']['nonnegative']:
        # Add the best fit nonnegative solution to the fit's `sol` object
        logging.info('  `nonnegative` is `true` in your parameter file --> Storing the best fit nonnegative profile as the attribute `nonneg` in the `sol` object')
        setattr(sol, '_nonneg', sol.solve_non_negative())

    logging.info('    Time taken to fit profile (with {:.0e} visibilities and'
                 ' {:d} collocation points) {:.1f} sec'.format(len(u),
                                                               model['hyperparameters']['n'],
                                                               time.time() - t1)
                 )

    if need_iterations:
        return sol, FF.iteration_diagnostics
    else:
        return [sol, None]


def run_multiple_fits(u, v, vis, weights, ncols, geom, model):
    r"""
    Perform and overplot multiple fits to a dataset by varying two of the
    model hyperparameters

    Parameters
    ----------
    u, v : array, unit = :math:`\lambda`
        u and v coordinates of observations
    vis : array, unit = Jy
        Observed visibilities (complex: real + imag * 1j)
    weights : array, unit = Jy^-2
        Weights assigned to observed visibilities, of the form
        :math:`1 / \sigma^2`
    ncols: int
        Number of columns in the UVTables to be saved
    geom : SourceGeometry object
        Fitted geometry (see frank.geometry.SourceGeometry)
    model : dict
        Dictionary containing model parameters the fits use

    Returns
    -------
    multifit_fig : Matplotlib `.Figure` instance
        All produced figures, including the GridSpecs
    multifit_axes : Matplotlib `~.axes.Axes` class
        Axes for each of the produced figures
    """

    logging.info(' Looping fits over the hyperparameters `alpha` and `wsmooth`')
    alphas = model['hyperparameters']['alpha']
    ws = model['hyperparameters']['wsmooth']
    sols = []

    def number_to_list(x):
        if np.isscalar(x):
            return [x]
        return x

    alphas = number_to_list(alphas)
    ws = number_to_list(ws)

    import copy
    for ii in range(len(alphas)):
        for jj in range(len(ws)):
            this_model = copy.deepcopy(model)
            this_model['hyperparameters']['alpha'] = alphas[ii]
            this_model['hyperparameters']['wsmooth'] = ws[jj]
            this_model['input_output']['save_prefix'] += '_alpha{}_wsmooth{}'.format(alphas[ii], ws[jj])

            logging.info('  Running fit for alpha = {}, wsmooth = {}'.format(alphas[ii], ws[jj]))

            sol, _ = perform_fit(u, v, vis, weights, geom, this_model)
            sols.append(sol)

            # Save the fit for the current choice of hyperparameter values
            output_results(u, v, vis, weights, ncols, sol, geom, this_model)

    multifit_fig, multifit_axes = make_figs.make_multifit_fig(u, v, vis, weights, sols,
                                                           model['plotting']['bin_widths'],
                                                           ['alpha', 'wsmooth'],
                                                           [alphas, ws],
                                                           model['plotting']['distance'],
                                                           model['plotting']['force_style'],
                                                           model['input_output']['save_prefix'],
                                                           )

    return multifit_fig, multifit_axes


def output_results(u, v, vis, weights, ncols, sol, geom, model, iteration_diagnostics=None):
    r"""
    Save datafiles of fit results; generate and save figures of fit results (see
    frank.io.save_fit, frank.make_figs)

    Parameters
    ----------
    u, v : array, unit = :math:`\lambda`
        u and v coordinates of observations
    vis : array, unit = Jy
        Observed visibilities (complex: real + imag * 1j)
    weights : array, unit = Jy^-2
        Weights assigned to observed visibilities, of the form
        :math:`1 / \sigma^2`
    ncols : int
        Number of columns to save in the UVTable
    sol : _HankelRegressor object
        Reconstructed profile using Maximum a posteriori power spectrum
        (see frank.radial_fitters.FrankFitter)
    geom : SourceGeometry object
        Fitted geometry (see frank.geometry.SourceGeometry)
    model : dict
        Dictionary containing model parameters the fit uses
    iteration_diagnostics : _HankelRegressor object, optional, default=None
        Diagnostics of the fit iteration
        (see radial_fitters.FrankFitter.fit)

    Returns
    -------
    figs : Matplotlib `.Figure` instance
        All produced figures, including the GridSpecs
    axes : Matplotlib `~.axes.Axes` class
        Axes for each of the produced figures
    """

    logging.info('  Plotting results')

    figs, axes = [], []

    if model['plotting']['deprojec_plot']:
        deproj_fig, deproj_axes = make_figs.make_deprojection_fig(u, v, vis, geom,
                                                         model['plotting']['force_style'],
                                                         model['input_output']['save_prefix']
                                                         )

        figs.append(deproj_fig)
        axes.append(deproj_axes)

    if model['plotting']['quick_plot']:
        quick_fig, quick_axes = make_figs.make_quick_fig(u, v, vis, weights, sol,
                                                         model['plotting']['bin_widths'],
                                                         model['plotting']['distance'],
                                                         model['plotting']['force_style'],
                                                         model['input_output']['save_prefix']
                                                         )

        figs.append(quick_fig)
        axes.append(quick_axes)

    if model['plotting']['full_plot']:
        full_fig, full_axes = make_figs.make_full_fig(u, v, vis, weights, sol,
                                                      model['plotting']['bin_widths'],
                                                      model['hyperparameters']['alpha'],
                                                      model['hyperparameters']['wsmooth'],
                                                      model['plotting']['gamma'],
                                                      model['plotting']['distance'],
                                                      model['plotting']['force_style'],
                                                      model['input_output']['save_prefix']
                                                      )

        figs.append(full_fig)
        axes.append(full_axes)

    if model['plotting']['diag_plot']:
        diag_fig, diag_axes, _ = make_figs.make_diag_fig(sol.r, sol.q,
                                                         iteration_diagnostics,
                                                         model['plotting']['iter_plot_range'],
                                                         model['plotting']['force_style'],
                                                         model['input_output']['save_prefix']
                                                         )

        figs.append(diag_fig)
        axes.append(diag_axes)

    if model['analysis']['compare_profile']:
        dat = np.genfromtxt(model['analysis']['compare_profile']).T

        if len(dat) not in [2,3,4]:
            raise ValueError("The file in your .json's `analysis` --> "
                             "`compare_profile` must have 2, 3 or 4 "
                             "columns: r [arcsec], I [Jy / sr], "
                             "negative uncertainty [Jy / sr] (optional), "
                             "positive uncertainty [Jy / sr] (optional, "
                             "assumed equal to negative uncertainty if not "
                             "provided).")

        r_clean, I_clean = dat[0], dat[1]
        if len(dat) == 3:
            lo_err_clean, hi_err_clean = dat[2], dat[2]
        elif len(dat) == 4:
            lo_err_clean, hi_err_clean = dat[2], dat[3]
        else:
            lo_err_clean, hi_err_clean = None, None
        clean_profile = {'r': r_clean, 'I': I_clean, 'lo_err': lo_err_clean,
                         'hi_err': hi_err_clean}

        mean_convolved = None
        if model['analysis']['clean_beam']['bmaj'] is not None:
            mean_convolved = utilities.convolve_profile(sol.r, sol.mean,
                                                        geom.inc, geom.PA,
                                                        model['analysis']['clean_beam'])

        clean_fig, clean_axes = make_figs.make_clean_comparison_fig(u, v, vis,
                                                                    weights, sol,
                                                                    clean_profile,
                                                                    model['plotting']['bin_widths'],
                                                                    model['plotting']['gamma'],
                                                                    mean_convolved,
                                                                    model['plotting']['distance'],
                                                                    model['plotting']['force_style'],
                                                                    model['input_output']['save_prefix']
                                                                    )

        figs.append(clean_fig)
        axes.append(clean_axes)

    io.save_fit(u, v, vis, weights, ncols, sol,
                model['input_output']['save_prefix'],
                model['input_output']['save_solution'],
                model['input_output']['save_profile_fit'],
                model['input_output']['save_vis_fit'],
                model['input_output']['save_uvtables'],
                model['input_output']['iteration_diag'],
                iteration_diagnostics,
                model['input_output']['format']
                )

    return figs, axes, model


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

    if (type(model['hyperparameters']['alpha']) or \
    type(model['hyperparameters']['wsmooth'])) is list:
        raise ValueError("For the bootstrap, both `alpha` and `wsmooth` in your "
                         "parameter file must be a float, not a list.")

    profiles_bootstrap = []

    if model['hyperparameters']['nonnegative']:
        logging.info('  `nonnegative` is `true` in your parameter file --> '
                     'The best fit nonnegative profile (rather than the mean '
                     'profile) will be saved and used to generate the bootstrap '
                     'figure')

    for trial in range(model['analysis']['bootstrap_ntrials']):
        logging.info(' Bootstrap trial {} of {}'.format(trial + 1,
                                                        model['analysis']['bootstrap_ntrials']))

        u_s, v_s, vis_s, w_s = utilities.draw_bootstrap_sample(
            u, v, vis, weights)

        sol, _ = perform_fit(u_s, v_s, vis_s, w_s, geom, model)

        if model['hyperparameters']['nonnegative']:
            profiles_bootstrap.append(sol._nonneg)
        else:
            profiles_bootstrap.append(sol.mean)

    bootstrap_path = model['input_output']['save_prefix'] + '_bootstrap.npz'

    logging.info(' Bootstrap complete. Saving fitted brightness profiles and'
                 ' the common set of collocation points')

    np.savez(bootstrap_path, r=sol.r, profiles=np.array(profiles_bootstrap))

    boot_fig, boot_axes = make_figs.make_bootstrap_fig(sol.r,
                                                       profiles_bootstrap,
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

    model, param_path = parse_parameters(*args)

    u, v, vis, weights, ncols = load_data(model)

    geom = determine_geometry(u, v, vis, weights, model)

    if model['modify_data']['baseline_range'] or \
            model['modify_data']['correct_weights']:
        u, v, vis, weights = alter_data(
            u, v, vis, weights, geom, model)

    if model['analysis']['bootstrap_ntrials']:
        boot_fig, boot_axes = perform_bootstrap(
            u, v, vis, weights, geom, model)

        return boot_fig, boot_axes

    elif (type(model['hyperparameters']['alpha']) or \
    type(model['hyperparameters']['wsmooth'])) is list:
        multifit_fig, multifit_axes = run_multiple_fits(u, v, vis, weights,
                                                        ncols, geom, model)

        return multifit_fig, multifit_axes

    else:
        sol, iteration_diagnostics = perform_fit(
            u, v, vis, weights, geom, model)

        figs, axes, model = output_results(u, v, vis, weights, ncols, sol, geom,
                                           model, iteration_diagnostics
                                           )

        logging.info('  Updating {} with final parameters used'
                     ''.format(param_path))
        with open(param_path, 'w') as f:
            json.dump(model, f, indent=4)

        logging.info("IT'S ALIVE!!\n")

        return figs, axes


if __name__ == "__main__":
    main()
