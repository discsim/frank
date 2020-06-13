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
"""This module runs tests to confirm Frankenstein is working correctly."""

import numpy as np
import os
import json

from frank.hankel import DiscreteHankelTransform
from frank.radial_fitters import FourierBesselFitter, FrankFitter
from frank.geometry import (
    FixedGeometry, FitGeometryGaussian, FitGeometryFourierBessel
)
from frank.constants import deg_to_rad
from frank.utilities import UVDataBinner
from frank.io import load_uvtable, save_uvtable
from frank import fit


def test_hankel_gauss():
    """Check the Hankel transform"""

    def gauss_real_space(r):
        x = r
        return np.exp(-0.5 * x * x)

    def gauss_vis_space(q):
        qs = (2 * np.pi) * q
        return np.exp(-0.5 * qs * qs) * (2 * np.pi)

    DHT = DiscreteHankelTransform(5.0, 100)

    Ir = gauss_real_space(DHT.r)
    Iq = gauss_vis_space(DHT.q)

    # Test at the DHT points.
    # Use a large error estimate because the DHT is approximate
    np.testing.assert_allclose(Iq, DHT.transform(Ir, direction='forward'),
                               atol=1e-5, rtol=0, err_msg="Forward DHT")

    np.testing.assert_allclose(Ir, DHT.transform(Iq, direction='backward'),
                               atol=1e-5, rtol=0, err_msg="Inverse DHT")

    # Test at generic points.
    # Larger error needed
    q = np.linspace(0.0, 1.0, 25)
    np.testing.assert_allclose(gauss_vis_space(q),
                               DHT.transform(Ir, q=q, direction='forward'),
                               atol=1e-4, rtol=0, err_msg="Generic Forward DHT")

    r = np.linspace(0, 5.0, 25)
    np.testing.assert_allclose(gauss_real_space(r),
                               DHT.transform(Iq, q=r, direction='backward'),
                               atol=1e-4, rtol=0, err_msg="Generic Inverse DHT")

    # Check the coefficients matrix works
    Hf = DHT.coefficients(direction='forward')
    Hb = DHT.coefficients(direction='backward')

    np.testing.assert_allclose(np.dot(Hf, Ir),
                               DHT.transform(Ir, direction='forward'),
                               rtol=1e-7, err_msg="Forward DHT Coeffs")
    np.testing.assert_allclose(np.dot(Hb, Iq),
                               DHT.transform(Iq, direction='backward'),
                               rtol=1e-7, err_msg="Inverse DHT Coeffs")

    # Compare cached vs non-cached DHT points
    np.testing.assert_allclose(DHT.coefficients(q=DHT.q),
                               DHT.coefficients(),
                               atol=1e-12, rtol=0,
                               err_msg="Cached forward DHT Coeffs"
                               )
    np.testing.assert_allclose(DHT.coefficients(q=DHT.r, direction='backward'),
                               DHT.coefficients(direction='backward'),
                               atol=1e-12, rtol=0,
                               err_msg="Cached inverse DHT Coeffs"
                               )


def test_import_data():
    """Check the UVTable import function works for a .txt"""
    load_uvtable('docs/tutorials/test_datafile.txt')


def load_AS209(uv_cut=None):
    """Load data for subsequent tests"""
    uv_AS209_DSHARP = np.load('docs/tutorials/AS209_continuum.npz')
    AS209_geometry = FixedGeometry(dRA=-1.9e-3, dDec=2.5e-3, inc=34.97,
                             PA=85.76)

    if uv_cut is not None:
        u, v = [uv_AS209_DSHARP[x] for x in ['u', 'v']]

        q = np.hypot(*AS209_geometry.deproject(u,v))

        keep = q < uv_cut

        cut_data = {}
        for key in  uv_AS209_DSHARP:
            if key not in { 'u', 'v', 'ReV', 'ImV', 'weights' }:
                continue
            cut_data[key] = uv_AS209_DSHARP[key][keep]

        uv_AS209_DSHARP = cut_data

    return uv_AS209_DSHARP, AS209_geometry


def test_fit_geometry():
    """Check the geometry fit on a subset of the AS209 data"""
    AS209, _ = load_AS209()
    u, v, re, im, weights = [AS209[k][::100] for k in ['u', 'v', 'ReV', 'ImV', 'weights']]
    vis = re + 1j * im

    geom = FitGeometryGaussian()
    geom.fit(u, v, vis, weights)

    np.testing.assert_allclose([geom.PA, geom.inc, 1e3 * geom.dRA,
                                1e3 * geom.dDec],
                               [1.4916013559412147 / deg_to_rad,
                                0.5395904796783955 / deg_to_rad,
                                -0.6431627790617276, -1.161768824369382],
                               err_msg="Gaussian geometry fit")

    geom = FitGeometryFourierBessel(1.6, 20)
    geom.fit(u, v, vis, weights)

    np.testing.assert_allclose([geom.PA, geom.inc, 1e3 * geom.dRA,
                               1e3 * geom.dDec],
                              [85.26142233422196, 33.819364762203364,
                               0.5611211784189547, -1.170097994325657],
                               rtol=1e-5,
                              err_msg="FourierBessel geometry fit")
    
def test_fourier_bessel_fitter():
    """Check FourierBesselFitter fitting routine with AS 209 dataset"""
    AS209, AS209_geometry = load_AS209()

    u, v, re, im, weights = [AS209[k] for k in ['u', 'v', 'ReV', 'ImV', 'weights']]
    vis = re + 1j * im

    Rmax = 1.6

    FB = FourierBesselFitter(Rmax, 20, geometry=AS209_geometry)

    sol = FB.fit(u, v, vis, weights)
    expected = np.array([1.89446696e+10, 1.81772972e+10, 1.39622125e+10,
                         1.20709653e+10,
                         9.83716859e+09, 3.26308106e+09, 2.02453146e+08,
                         4.73919867e+09,
                         1.67911877e+09, 1.73161931e+08, 4.50233539e+08,
                         3.57108466e+08,
                         4.04216831e+09, 1.89085113e+09, 6.73819228e+08,
                         5.50895976e+08,
                         1.53683576e+08, 1.02413038e+08, 2.32589333e+07,
                         3.33260713e+07
                         ])

    np.testing.assert_allclose(sol.mean, expected,
                               err_msg="Testing FourierBessel Fit to AS 209")


def test_frank_fitter():
    """Check FrankFitter fitting routine with AS 209 dataset"""
    AS209, AS209_geometry = load_AS209(uv_cut=1e6)

    u, v, re, im, weights = [AS209[k][::100] for k in ['u', 'v', 'ReV', 'ImV', 'weights']]
    vis = re + 1j * im

    Rmax = 1.6

    FF = FrankFitter(Rmax, 20, AS209_geometry, alpha=1.05, weights_smooth=1e-2)

    sol = FF.fit(u, v, vis, weights)
    print(sol.mean)
    expected = np.array([
        1.98412226e+10,  1.88897931e+10,  1.35298932e+10,  1.25937780e+10,
        1.06247023e+10,  2.41041917e+09,  8.48900947e+08,  4.25620911e+09,
        2.19599281e+09,  5.53184738e+07,  4.65888337e+07,  9.73402488e+08,
        3.56565340e+09,  2.52916177e+09,  3.39299547e+08,  6.87694292e+08,
        2.23703869e+08,  1.36913099e+08,  2.55069045e+06, -1.10932464e+08        
    ])

    np.testing.assert_allclose(sol.mean, expected,
                               err_msg="Testing Frank Fit to AS 209")


def test_fit_geometry_inside():
    """Check the geometry fit embedded in a call to FrankFitter"""
    AS209, _ = load_AS209(uv_cut=1e6)

    u, v, re, im, weights = [AS209[k][::100] for k in ['u', 'v', 'ReV', 'ImV', 'weights']]
    vis = re + 1j * im

    Rmax = 1.6

    FF = FrankFitter(Rmax, 20, FitGeometryGaussian(),
                     alpha=1.05, weights_smooth=1e-2)

    sol = FF.fit(u, v, vis, weights)

    geom = sol.geometry
    np.testing.assert_allclose([geom.PA, geom.inc, 1e3 * geom.dRA,
                                1e3 * geom.dDec],
                               [86.46568992560152, 34.5071920284988,
                                0.20818634201418384, -2.0988159662202714],
                               err_msg="Gaussian geometry fit inside Frank fit")

def test_throw_error_on_bad_q_range():
    """Check that frank correctly raises an error when the
    q range is bad."""
    AS209, AS209_geometry = load_AS209()

    u, v, re, im, weights = [AS209[k][::100] for k in ['u', 'v', 'ReV', 'ImV', 'weights']]
    vis = re + 1j * im
    
    Rmax = 1.6

    FF = FrankFitter(Rmax, 20, AS209_geometry,
                     alpha=1.05, weights_smooth=1e-2)

    try:
        sol = FF.fit(u, v, vis, weights)
        raise RuntimeError("Expected ValueError due to bad range")
    except ValueError:
        pass

def test_uvbin():
    """Check the uv-data binning routine"""
    AS209, AS209_geometry = load_AS209()

    u, v, re, im, weights = [AS209[k][::100] for k in ['u', 'v', 'ReV', 'ImV', 'weights']]
    vis = re + 1j * im
    
    uv = np.hypot(*AS209_geometry.deproject(u, v))

    uvbin = UVDataBinner(uv, vis, weights, 50e3)

    uvmin = 1e6
    uvmax = 1e6 + 50e3

    idx = (uv >= uvmin) & (uv < uvmax)

    widx = weights[idx]

    w = np.sum(widx)
    V = np.sum(widx*vis[idx]) / w
    q = np.sum(widx*uv[idx]) / w

    i = (uvbin.uv >= uvmin) & (uvbin.uv < uvmax)

    np.testing.assert_allclose(q, uvbin.uv[i])
    np.testing.assert_allclose(V, uvbin.V[i])
    np.testing.assert_allclose(w, uvbin.weights[i])
    np.testing.assert_allclose(len(widx), uvbin.bin_counts[i])


def _run_pipeline(geometry='gaussian', fit_phase_offset=True, make_figs=False, 
                   multifit=False, bootstrap=False):
    """Check the full pipeline that performs a fit and outputs results"""

    # Build a subset of the data that we'll load during the fit
    AS209, AS209_geometry = load_AS209(uv_cut=1e6)

    u, v, re, im, weights = [AS209[k][::100] for k in ['u', 'v', 'ReV', 'ImV', 'weights']]
    vis = re + 1j * im
    ncols = 5
    
    tmp_dir = '/tmp/frank/tests'
    os.makedirs(tmp_dir, exist_ok=True)

    uv_table = os.path.join(tmp_dir, 'small_uv.npz')
    save_uvtable(uv_table, u, v, vis, weights, ncols)

    # Build a parameter file
    params = fit.load_default_parameters()

    params['input_output']['uvtable_filename'] = uv_table

    # Set the model parameters
    params['hyperparameters']['n'] = 20
    params['hyperparameters']['rout'] = 1.6
    params['hyperparameters']['alpha'] = 1.05
    params['hyperparameters']['wmsooth'] = 1e-2

    geom = params['geometry']
    geom['type'] = geometry
    geom['fit_phase_offset'] = fit_phase_offset
    geom['inc'] = AS209_geometry.inc
    geom['pa'] = AS209_geometry.PA
    geom['dra'] = AS209_geometry.dRA
    geom['ddec'] = AS209_geometry.dDec

    if make_figs:
        params['plotting']['quick_plot'] = True
        params['plotting']['full_plot'] = True
        params['plotting']['diag_plot'] = True
        params['plotting']['deprojec_plot'] = True
        params['plotting']['save_figs'] = True
        params['plotting']['distance'] = 121.
        params['plotting']['bin_widths'] = [1e5]
        params['plotting']['iter_plot_range'] = [0, 5]
        params['analysis']['compare_profile'] = 'docs/tutorials/AS209_clean_profile.txt'
        params['analysis']['clean_beam'] = {'bmaj'    : 0.03883,
                                            'bmin'    : 0.03818,
                                            'beam_pa' : 85.82243
                                            }

    else:
        params['plotting']['quick_plot'] = False
        params['plotting']['full_plot'] = False
        params['plotting']['diag_plot'] = False
        params['plotting']['deprojec_plot'] = False
    
    if multifit: 
        params['hyperparameters']['alpha'] = [1.05, 1.30]        
        
    if bootstrap:
        params['analysis']['bootstrap_ntrials'] = 2
    
    # Save the new parameter file
    param_file = os.path.join(tmp_dir, 'params.json')
    with open(param_file, 'w') as f:
        json.dump(params, f)

    # Call the pipeline to perform the fit
    fit.main(['-p', param_file])


def test_pipeline_full_geom():
    """Check the full fit pipeline when fitting for the disc's inc, PA, dRA, dDec"""
    _run_pipeline('gaussian', True)


def test_pipeline_no_phase():
    """Check the full fit pipeline when fitting for the disc's inc, PA"""
    _run_pipeline('gaussian', False)


def test_pipeline_known_geom():
    """Check the full fit pipeline when supplying a known disc geometry"""
    _run_pipeline('known')


def test_pipeline_figure_generation():
    """Check the full fit pipeline when producing all figures"""
    _run_pipeline('known', make_figs=True)
    
    
def test_pipeline_multifit():
    """Check the full fit pipeline when producing all figures"""
    _run_pipeline('known', multifit=True)    
    
    
def test_pipeline_bootstrap():
    """Check the full fit pipeline when producing all figures"""
    _run_pipeline('known', bootstrap=True)        
