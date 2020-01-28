"""Run Frankenstein to fit a source's 1D radial brightness profile.
   A default parameter file is used that specifies all options to run the fit
   and output results. Alternatively a custom parameter file can be provided.
"""

import sys
import argparse
import json

default_parameters = "default_parameters.json"


def help():
    param_descrip = {
      "input_output" : {
        "uvtable_filename" : "UV table with data to be fit. (columns: u, v, Re(V), Im(V), weights)",
        "load_dir" : "Directory containing UV table",
        "save_dir" : "Directory in which output datafiles and figures are saved",
        "save_profile_fit" : "Whether to save fitted brightness profile",
        "save_vis_fit" : "Whether to save fitted visibility distribution",
        "save_uvtables" : "Whether to save fitted and residual UV tables (these are reprojected)",
        "make_plots" : "Whether to make figures showing the fit and diagnostics",
        "save_plots" : "Whether to save figures",
        "dist" : "Distance to source, optionally used for plotting. [AU]",
      },

      "modify_data" : {
        "cut_data"  : "Whether to truncate the visibilities at a given maximum baseline prior to fitting",
        "cut_baseline"  : "Maximum baseline at which visibilities are truncated",
      },

      "geometry" : {
        "deproject_data" : "Whether to correct the visibilities for on-sky projection",
        "known_geometry" : "Whether to manually specify a geometry (if False, geometry will be fitted)",
        "inc" : "Inclination. [deg]",
        "pa" : "Position angle. [deg]",
        "dra" : "Delta (offset from 0) right ascension. [arcsec]",
        "ddec" : "Delta declination. [arcsec]",
      },

      "hyperpriors" : {
        "n" : "Number of collocation points used in the fit (suggested range 100 - 300)",
        "rout" : "Maximum disc radius in the fit (best to overestimate size of source). [arcsec]",
        "alpha" : "Order parameter for the power spectrum's inverse Gamma prior (suggested range 1.00 - 1.50)",
        "p0" : "Scale parameter for the power spectrum's inverse Gamma prior (suggested >0, <<1)",
        "wsmooth" : "Strength of smoothing applied to the power spectrum (suggested range 10^-4 - 10^-1)",
      }
    }

    print("""
     Fit a 1D radial brightness profile with Frankenstein from the \
     terminal with `python fit.py`. A .json parameter file is required. The
     default is default_parameters.json and is of the form:\n\n""",
     json.dumps(param_descrip, indent=4))


def parse_parameters(parameter_file):
    parser = argparse.ArgumentParser("Run a Frank fit, by default using parameters in default_parameters.json")
    parser.add_argument("-p", "--parameters", default=default_parameters, type=str, help="Parameter file (.json)")

    args = parser.parse_args()
    model = json.load(open(args.parameters, 'r'))

    return args, model


def convert_units(model):
    rout = model['hyperpriors']['rout'] / rad_to_arcsec

    return rout


def load_uvdata(data_file):
    u, v, vis, weights = np.genfromtxt(data_file).T

    return u, v, vis, weights


def deproject_disc(model, u, v, vis):



def perform_fit(model, rout):
    from frank.radial_fitters import FrankFitter

    FF = FrankFitter(rout, model['hyperpriors']['n'], geometry=geometry, \
         alpha=model['hyperpriors']['alpha'], weights_smooth=model['hyperpriors']['wsmooth'])

    sol = FF.fit(u, v, vis, weights)

    return sol


def output_results(model, u, v, vis, weights, sol):
    if model['input_output']['save_profile_fit']:
        np.savetxt(savedir + 'fit.txt', np.array([sol.r, sol.mean, np.diag(sol.covariance)**.5]).T, \
            header='r [arcsec]\tI [Jy/sr]\tI_err [Jy/sr]'))

    if model['input_output']['save_vis_fit']:
        np.savetxt(savedir + 'fit_vis.txt', np.array([ki / (2 * np.pi), GPHF.HankelTransform(ki)]).T, \
            header='Baseline [lambda]\tRe(V) [Jy]')

    if model['input_output']['save_uvtables']:
        np.save(savedir + disc + '_frank_fit.dat', np.stack([u_proj, v_proj, re_proj, im_proj, weights_orig], axis=-1))
        np.save(savedir + disc + '_frank_residuals.dat', np.stack([u_proj, v_proj, re_proj, im_proj, weights_orig], axis=-1))

    if model['input_output']['make_plots']:
        frank.plot(model, u, v, vis, weights, sol)
        if model['input_output']['save_plots']:
            frank.save_plot(fn)


def main():
    args, model = parse_parameters(parameter_file)

    rout = convert_units(model)

    u, v, vis, weights = load_uvdata(data_file)

    u, v, vis = deproject_disc(model, u, v, vis, weights)

    perform_fit(model, u, v, vis, weights)

    output_results(model, u, v, vis, weights, sol)

    print("\n\nIT'S ALIVE!!\n\n")


if __name__ == "__main__":
    main()
