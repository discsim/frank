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
    """

    u, v, vis, weights = io.load_uvtable(
        model['input_output']['uvtable_filename'])

    return u, v, vis, weights


def alter_data(u, v, vis, weights, geometry, model):
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
    geometry : SourceGeometry object
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
                                           geometry)

    wcorr_estimate = None
    if model['modify_data']['correct_weights']:
        up, vp = geometry.deproject(u,v)
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

    elif model['geometry']['type'] == 'gaussian':
        t1 = time.time()

        if model['geometry']['fit_phase_offset']:
            geom = geometry.FitGeometryGaussian()

        else:
            geom = geometry.FitGeometryGaussian(phase_centre=(model['geometry']['dra'],
                                                              model['geometry']['ddec']))

        geom.fit(u, v, vis, weights)

        logging.info('    Time taken for geometry %.1f sec' %
                     (time.time() - t1))

    else:
        raise ValueError("geometry_type in your parameter file must be one of"
                         " 'known' or 'gaussian'")

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
        # Replace the mean solution with the best fit non-negative
        sol.mean = sol.solve_non_negative()

    logging.info('    Time taken to fit profile (with {:.0e} visibilities and'
                 ' {:d} collocation points) {:.1f} sec'.format(len(u),
                                                               model['hyperparameters']['n'],
                                                               time.time() - t1)
                 )

    if need_iterations:
        return sol, FF.iteration_diagnostics
    else:
        return [sol, None]


def output_results(u, v, vis, weights, sol, model, iteration_diagnostics=None):
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

    if model['plotting']['quick_plot']:
        quick_fig, quick_axes = make_figs.make_quick_fig(u, v, vis, weights, sol,
                                                         model['plotting']['bin_widths'],
                                                         model['plotting']['dist'],
                                                         model['plotting']['force_style'],
                                                         model['input_output']['save_prefix']
                                                         )

        figs.append(quick_fig)
        axes.append(quick_axes)

    if model['plotting']['full_plot']:
        full_fig, full_axes = make_figs.make_full_fig(u, v, vis, weights, sol,
                                                      model['plotting']['bin_widths'],
                                                      [model['hyperparameters']['alpha'],
                                                       model['hyperparameters']['wsmooth']],
                                                      model['plotting']['dist'],
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

    io.save_fit(u, v, vis, weights, sol,
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


def run_multiple_fits(u, v, vis, weights, geom, model):
    r"""
    Perform and overplot multiple fits to a dataset by varying two of the
    model hyperparameters)

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
        Dictionary containing model parameters the fits use

    Returns
    -------
    multifit_fig : Matplotlib `.Figure` instance
        All produced figures, including the GridSpecs
    multifit_axes : Matplotlib `~.axes.Axes` class
        Axes for each of the produced figures
    """

    hpars = list(model['hyperparameters'].values())
    multis_bool = [isinstance(x, list) for i, x in enumerate(
        hpars)]
    multis_idx = [i for i, x in enumerate(multis_bool) if x]
    if len(multis_idx) != 2:
        raise ValueError("run_multiple_fits supports varying only two"
                         " hyperparameters; please reduce the number of"
                         " lists in `hyperparameters` in your parameter file")

    multis_hpar0 = list(model['hyperparameters'].keys())[multis_idx[0]]
    multis_hpar1 = list(model['hyperparameters'].keys())[multis_idx[1]]
    multis_vals0 = hpars[multis_idx[0]]
    multis_vals1 = hpars[multis_idx[1]]

    logging.info(' Looping fits over the hyperparameters {} and {}'.format(multis_hpar0, multis_hpar1))

    sols = []
    used_vals0 = []
    used_vals1 = []
    import copy
    for ii in range(len(multis_vals0)):
        for jj in range(len(multis_vals1)):
            this_model = copy.deepcopy(model)
            this_model['hyperparameters']['{}'.format(multis_hpar0)] = multis_vals0[ii]
            this_model['hyperparameters']['{}'.format(multis_hpar1)] = multis_vals1[jj]
            this_model['input_output']['save_prefix'] = model['input_output']['save_prefix'] + '_{}{}_{}{}'.format(multis_hpar0, multis_vals0[ii], multis_hpar1, multis_vals1[jj])
            used_vals0.append(multis_vals0[ii])
            used_vals1.append(multis_vals1[jj])

            logging.info('  Running fit for {} = {}, {} = {}'.format(multis_hpar0, multis_vals0[ii], multis_hpar1, multis_vals1[jj]))

            sol, iteration_diagnostics = perform_fit(u, v, vis, weights, geom, this_model)
            sols.append(sol)

            # Save the fit for the current choice of hyperparameter values
            output_results(u, v, vis, weights, sol, this_model)

    multifit_fig, multifit_axes = make_figs.make_multifit_fig(u, v, vis, weights, sols,
                                                           model['plotting']['bin_widths'],
                                                           [multis_hpar0, multis_hpar1],
                                                           [used_vals0, used_vals1],
                                                           model['plotting']['dist'],
                                                           model['plotting']['force_style'],
                                                           model['input_output']['save_prefix'],
                                                           )

    return multifit_fig, multifit_axes


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

    for trial in range(model['analysis']['bootstrap_ntrials']):
        logging.info(' Bootstrap trial {} of {}'.format(trial + 1,
                                                        model['analysis']['bootstrap_ntrials']))

        u_s, v_s, vis_s, w_s = utilities.draw_bootstrap_sample(
            u, v, vis, weights)

        sol, _ = perform_fit(u_s, v_s, vis_s, w_s, geom, model)

        profiles_bootstrap.append(sol.mean)

    bootstrap_path = model['input_output']['save_prefix'] + '_bootstrap.npz'

    logging.info(' Bootstrap complete. Saving fitted brightness profiles and'
                 ' the common set of collocation points')

    np.savez(bootstrap_path, r=sol.r, profiles=np.array(profiles_bootstrap))

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

    model, param_path = parse_parameters(*args)

    u, v, vis, weights = load_data(model)

    geom = determine_geometry(u, v, vis, weights, model)

    if model['modify_data']['baseline_range'] or \
            model['modify_data']['correct_weights']:
        u, v, vis, weights = alter_data(
            u, v, vis, weights, geom, model)

    if model['analysis']['bootstrap_ntrials']:
        boot_fig, boot_axes = perform_bootstrap(
            u, v, vis, weights, geom, model)

        return boot_fig, boot_axes

    elif any(isinstance(x, list) for x in model['hyperparameters'].values()):
        multifit_fig, multifit_axes = run_multiple_fits(u, v, vis, weights,
                                                        geom, model)

        return multifit_fig, multifit_axes

    else:
        sol, iteration_diagnostics = perform_fit(
            u, v, vis, weights, geom, model)

        figs, axes, model = output_results(u, v, vis, weights, sol, model,
                                           iteration_diagnostics
                                           )

        logging.info('  Updating {} with final parameters used'
                     ''.format(param_path))
        with open(param_path, 'w') as f:
            json.dump(model, f, indent=4)

    logging.info("IT'S ALIVE!!\n")

    return figs, axes


if __name__ == "__main__":
    main()
