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
"""Run Frankenstein to fit a source's 1D radial brightness profile.
   A default parameter file is used that specifies all options to run the fit
   and output results. Alternatively a custom parameter file can be provided.
"""

import os
import sys
import time
import json
import numpy as np

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[
                    logging.FileHandler('frank_fit.log', mode='w'), # TODO: want to save this in `save_dir`...how to?
                    logging.StreamHandler()])


def helper():
    with open('parameter_descriptions.json') as f:
        param_descrip = json.load(f) # TODO: point to parameter_descriptions.json in code dir

    print("""
     Fit a 1D radial brightness profile with Frankenstein (frank) from the
     terminal with `python -m frank.fit`. A .json parameter file is required;
     the default is default_parameters.json and is of the form:\n\n""",
     json.dumps(param_descrip, indent=4))


def parse_parameters():
    """
    Read in a .json parameter file to set the fit parameters.

    Parameters
    ----------
    parameter_filename : string
            Parameter file (.json; see frank.fit.helper).
            Defaults to `default_parameters.json`
    uvtable_filename : string
            UVTable file with data to be fit (.txt). The UVTable column format
            should be u [lambda]  v [lambda] Re(V) [Jy]  Im(V) [Jy]
            Weight [Jy^-2]

    Returns
    -------
    model : dict
            Dictionary containing model parameters the fit uses
    """

    import argparse

    parser = argparse.ArgumentParser("Run a Frank fit, by default using"
                                     " parameters in default_parameters.json")
    parser.add_argument("-p", "--parameter_filename",
                        default='default_parameters.json', type=str,
                        help="Parameter file (.json; see frank.fit.helper)") # TODO: redundant to list this above in docstring?
    parser.add_argument("-uv", "--uvtable_filename", default=None, type=str,
                        help="UVTable file with data to be fit (.txt). The" # TODO: redundant to list this above in docstring?
                        " UVTable column format should be u [lambda]  v [lambda]"
                        " Re(V) [Jy]  Im(V) [Jy]  Weight [Jy^-2]")

    args = parser.parse_args()
    model = json.load(open(args.parameters, 'r'))

    if args.uvtable_filename:
        model['input_output']['uvtable_filename'] = args.uvtable_filename

    if ('uvtable_filename' not in model['input_output'] or
        not model['input_output']['uvtable_filename']):
        raise ValueError("    uvtable_filename isn't specified."
                 " Set it in the parameter file or run frank with"
                 " python -m frank.fit -uv <uvtable_filename>")

    if not model['input_output']['load_dir']:
        model['input_output']['load_dir'] = os.getcwd()

    if not model['input_output']['save_dir']:
        model['input_output']['save_dir'] = model['input_output']['load_dir']

    logging.info('\nRunning frank on %s'%model['input_output']['uvtable_filename'])

    logging.info('  Saving parameters to be used in fit to `frank_used_pars.json`')
    with open('frank_used_pars.json', 'w') as f:
        json.dump(model, f, indent=4)

    return model


def load_uvdata(data_file):
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

    logging.info('  Loading UVTable')

    u, v, vis, weights = np.genfromtxt(data_file).T
    # TODO: (optionally) convert u, v from [m] to [lambda]

    return u, v, vis, weights


def determine_geometry(model, u, v, vis, weights):
    """
    Determine the source geometry (inclination, position angle, phase offset).

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

    Returns
    -------
    geom : SourceGeometry object
          Fitted geometry (see frank.geometry.SourceGeometry)
    """

    logging.info('  Determining disc geometry')

    if not model['geometry']['fit_geometry']:
        from frank.geometry import FixedGeometry
        geom = FixedGeometry(0., 0., 0., 0.)

    else:
        if model['geometry']['known_geometry']:
            from frank.geometry import FixedGeometry

            geom = FixedGeometry(model['geometry']['inc'],
                                 model['geometry']['pa'],
                                 model['geometry']['dra'],
                                 model['geometry']['ddec']
                                 )

        else:
            from frank.geometry import FitGeometryGaussian

            if model['geometry']['fit_phase_offset']:
                geom = FitGeometryGaussian()

            else:
                geom = FitGeometryGaussian(phase_centre=(model['geometry']['dra'],
                                           model['geometry']['ddec']))

            t1 = time.time()
            geom.fit(u, v, vis, weights)
            logging.info('    Time taken to fit geometry %.1f sec'%(time.time() - t1))

    logging.info('    Using: inc  = %.2f deg,\n           PA   = %.2f deg,\n'
          '           dRA  = %.2e arcsec,\n           dDec = %.2e arcsec'
          %(geom.inc, geom.PA, geom.dRA, geom.dDec))

    return geom


def perform_fit(model, u, v, vis, weights, geom):
    """
    Deproject the observed visibilities and fit them for the brightness profile.

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
    geom : SourceGeometry object
          Fitted geometry (see frank.geometry.SourceGeometry)

    Returns
    -------
    sol : _HankelRegressor object
          Reconstructed profile using Maximum a posteriori power spectrum
          (see frank.radial_fitters.FrankFitter) # TODO: check
    """

    logging.info('  Fitting for brightness profile')

    from frank.radial_fitters import FrankFitter

    FF = FrankFitter(Rmax=model['hyperpriors']['rout'],
                     N=model['hyperpriors']['n'],
                     geometry=geom,
                     alpha=model['hyperpriors']['alpha'],
                     weights_smooth=model['hyperpriors']['wsmooth']
                     )

    t1 = time.time()
    sol = FF.fit(u, v, vis, weights)
    logging.info('    Time taken to fit profile (with %.0e visibilities and %s'
          ' collocation points) %.1f sec'%(len(vis), model['hyperpriors']['n'],
          time.time() - t1))

    return sol


def output_results(model, u, v, vis, weights, geom, sol, diag_fig=True):
    """
    Save datafiles of fit results; generate and save figures of fit results.

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
    geom : SourceGeometry object
          Fitted geometry (see frank.geometry.SourceGeometry)
    sol : _HankelRegressor object
          Reconstructed profile using Maximum a posteriori power spectrum
          (see frank.radial_fitters.FrankFitter) # TODO: check
    diag_fig : bool, optional, default=True
        Whether to produce a figure showing diagnostics of the fit
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
                   header='Baseline [lambda]\tProjected Re(V) [Jy]') # TODO: change to match data (deprojected?) baselines

    import matplotlib.pyplot as plt
    from frank.constants import deg_to_rad
    plt.figure()
    plt.loglog(np.hypot(u,v), vis.real, 'k.')
    plt.loglog(np.hypot(u,v), sol.predict(u,v).real,'g.')
    plt.loglog(sol.q, sol.predict_deprojected(sol.q).real, 'r.')
    plt.savefig(prefix + '_test.png')

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

    if model['input_output']['make_plots']:
        from frank.plot import plot_fit
        plot_fit(model, u, v, vis, weights, geom, sol, diag_fig,
                 model['input_output']['save_plots'])


def main():
    model = parse_parameters()

    u, v, vis, weights = load_uvdata(model['input_output']['uvtable_filename'])

    geom = determine_geometry(model, u, v, vis, weights)

    sol = perform_fit(model, u, v, vis, weights, geom)

    output_results(model, u, v, vis, weights, geom, sol)

    logging.info("IT'S ALIVE!!\n")

if __name__ == "__main__":
    main()
