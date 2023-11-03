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

from frank.constants import rad_to_arcsec
from frank.hankel import DiscreteHankelTransform
from frank.radial_fitters import FourierBesselFitter, FrankFitter
from frank.debris_fitters import FrankDebrisFitter
from frank.geometry import (
    FixedGeometry, FitGeometryGaussian, FitGeometryFourierBessel
)
from frank import utilities
from frank.io import load_uvtable, save_uvtable, load_sol, save_fit
from frank.statistical_models import VisibilityMapping
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


def test_vis_mapping():
    def gauss_real_space(r):
        x = r 
        return np.exp(-0.5 * x * x)

    def gauss_vis_space(q):
        qs = (2 * np.pi) * q / rad_to_arcsec
        return np.exp(-0.5 * qs * qs) * (2 * np.pi / rad_to_arcsec**2)

    DHT = DiscreteHankelTransform(5.0/rad_to_arcsec, 100)
    geometry = FixedGeometry(60, 0, 0, 0)

    VM = VisibilityMapping(DHT, geometry)

    Ir = gauss_real_space(VM.r)
    Iq = gauss_vis_space(VM.q)

    # Test at the DHT points.
    # Use a large error estimate because the DHT is approximate
    np.testing.assert_allclose(Iq, 2*VM.predict_visibilities(Ir, VM.q),
                               atol=1e-5, rtol=0, err_msg="Forward DHT with VisibilityMapping")

    np.testing.assert_allclose(Ir, 0.5*VM.invert_visibilities(Iq, VM.r),
                               atol=1e-5, rtol=0, err_msg="Inverse DHT with VisibilityMapping")

    # Test generic_dht, which uses these functions
    _, Iq_dht = utilities.generic_dht(VM.r, Ir, inc=60, Rmax=5, N=100)
    _, Ir_dht = utilities.generic_dht(VM.q, Iq, inc=60, Rmax=5, N=100, direction='backward')

    np.testing.assert_allclose(Iq, 2*Iq_dht,
                               atol=1e-5, rtol=0, err_msg="Forward DHT with generic_dht")

    np.testing.assert_allclose(Ir, 0.5*Ir_dht,
                               atol=1e-5, rtol=0, err_msg="Inverse DHT with generic_dht")


def test_import_data():
    """Check the UVTable import function works for a .txt"""
    load_uvtable('docs/tutorials/test_datafile.txt')


def load_AS209(uv_cut=None):
    """Load data for subsequent tests"""
    uv_AS209_DSHARP = np.load('docs/tutorials/AS209_continuum.npz')
    geometry = FixedGeometry(dRA=-1.9e-3, dDec=2.5e-3, inc=34.97,
                             PA=85.76)

    if uv_cut is not None:
        u, v = [uv_AS209_DSHARP[x] for x in ['u', 'v']]

        q = np.hypot(*geometry.deproject(u,v))

        keep = q < uv_cut

        cut_data = {}
        for key in  uv_AS209_DSHARP:
            if key not in { 'u', 'v', 'V', 'weights' }:
                continue
            cut_data[key] = uv_AS209_DSHARP[key][keep]

        uv_AS209_DSHARP = cut_data

    return uv_AS209_DSHARP, geometry

def test_fit_geometry():
    """Check the geometry fit on a subset of the AS209 data"""
    AS209, _ = load_AS209()
    u, v, vis, weights = [AS209[k][::100] for k in ['u', 'v', 'V', 'weights']]

    inc_pa = [30.916257151011674, 85.46246241142246]
    phase_centre = [-0.6431627790617276e-3, -1.161768824369382e-3]

    geom = FitGeometryGaussian()
    geom.fit(u, v, vis, weights)

    np.testing.assert_allclose([geom.inc, geom.PA, 1e3 * geom.dRA,
                                1e3 * geom.dDec],
                               [30.916256948647096,
                                85.46246845532691,
                                -0.6434703241180601, -1.1623515516661052],
                               err_msg="Gaussian geometry fit")

    geom = FitGeometryGaussian(inc_pa=inc_pa)
    geom.fit(u, v, vis, weights)

    np.testing.assert_allclose([geom.inc, geom.PA, 1e3 * geom.dRA,
                                1e3 * geom.dDec],
                               [30.916257151011674,
                                85.46246241142246,
                                -0.6432951224590862, -1.1619271783674576],
                               err_msg="Gaussian geometry fit (provided inc_pa)")

    geom = FitGeometryGaussian(phase_centre=phase_centre)
    geom.fit(u, v, vis, weights)

    np.testing.assert_allclose([geom.inc, geom.PA, 1e3 * geom.dRA,
                                1e3 * geom.dDec],
                               [30.91601671340713,
                                85.471787339838,
                                -0.6431627790617276, -1.161768824369382],
                               err_msg="Gaussian geometry fit (provided phase_centre)")

    geom = FitGeometryGaussian(guess=[1.0, 1.0, 0.1, 0.1])
    geom.fit(u, v, vis, weights)

    np.testing.assert_allclose([geom.inc, geom.PA, 1e3 * geom.dRA,
                                1e3 * geom.dDec],
                                [30.91625882521282, 85.46246494153092,
                                 -0.6440453613101292, -1.1622414671266803],
                               err_msg="Gaussian geometry fit (provided guess)")

    geom = FitGeometryFourierBessel(1.6, 20)
    geom.fit(u, v, vis, weights)

    np.testing.assert_allclose([geom.inc, geom.PA, 1e3 * geom.dRA,
                               1e3 * geom.dDec],
                              [33.81936473347169, 85.26142233735665,
                               0.5611211784189547, -1.170097994325657],
                               rtol=1e-5,
                              err_msg="FourierBessel geometry fit")


    geom = FitGeometryFourierBessel(1.6, 20, inc_pa=inc_pa)
    geom.fit(u, v, vis, weights)

    np.testing.assert_allclose([geom.inc, geom.PA, 1e3 * geom.dRA,
                               1e3 * geom.dDec],
                              [30.916257151011674, 85.46246241142246,
                               1.4567005881700168, -1.658896248809076],
                               rtol=1e-5,
                              err_msg="FourierBessel geometry fit (provided inc_pa)")


    geom = FitGeometryFourierBessel(1.6, 20, phase_centre=phase_centre)
    geom.fit(u, v, vis, weights)

    np.testing.assert_allclose([geom.inc, geom.PA, 1e3 * geom.dRA,
                               1e3 * geom.dDec],
                              [33.83672738960381, 85.2562498368987,
                               -0.6431627790617276, -1.161768824369382],
                               rtol=1e-5,
                              err_msg="FourierBessel geometry fit (provided phase_centre)")


def test_fourier_bessel_fitter():
    """Check FourierBesselFitter fitting routine with AS 209 dataset"""
    AS209, geometry = load_AS209()

    u, v, vis, weights = [AS209[k] for k in ['u', 'v', 'V', 'weights']]

    Rmax = 1.6

    FB = FourierBesselFitter(Rmax, 20, geometry=geometry)

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

    np.testing.assert_allclose(sol.I, expected,
                               err_msg="Testing FourierBessel Fit to AS 209")


def test_frank_fitter():
    """Check FrankFitter fitting routine with AS 209 dataset"""
    AS209, geometry = load_AS209(uv_cut=1e6)

    u, v, vis, weights = [AS209[k] for k in ['u', 'v', 'V', 'weights']]

    Rmax = 1.6

    FF = FrankFitter(Rmax, 20, geometry, alpha=1.05, weights_smooth=1e-2)

    sol = FF.fit(u, v, vis, weights)
    expected = np.array([
         2.007570578977623749e+10,  1.843513239499150085e+10,
         1.349013094584499741e+10,  1.272363099855433273e+10,
         1.034881472041586494e+10,  2.579145371666701317e+09,
         6.973651829234187603e+08,  4.127687040627769947e+09,
         2.502124003048851490e+09, -2.756950827897560596e+08,
         2.823720459944381118e+08,  8.705940396227477789e+08,
         3.257425109322027683e+09,  3.112003905406182289e+09,
        -5.145431577819123268e+08,  1.491165153547247887e+09,
        -5.190942564982021451e+08,  5.100334030941848755e+08,
        -1.922568182176418006e+08,  8.067782715878820419e+07,
    ])

    np.testing.assert_allclose(sol.I, expected,
                               err_msg="Testing Frank Fit to AS 209")


def test_two_stage_fit():
    """Check FrankFitter fitting routine with AS 209 dataset"""
    AS209, geometry = load_AS209(uv_cut=1e6)

    u, v, vis, weights = [AS209[k] for k in ['u', 'v', 'V', 'weights']]

    Rmax = 1.6

    FF = FrankFitter(Rmax, 20, geometry, alpha=1.05, weights_smooth=1e-2)

    # 1 step fit
    sol = FF.fit(u, v, vis, weights)

    # 2 step fit
    preproc = FF.preprocess_visibilities(u, v, vis, weights)
    sol2 = FF.fit_preprocessed(preproc)

    np.testing.assert_equal(sol.I, sol2.I,
                            err_msg="Testing two-step fit")


def test_solve_non_negative():
    """Check FrankFitter fitting routine with non-negative fit using AS 209 dataset"""
    AS209, geometry = load_AS209(uv_cut=1e6)

    u, v, vis, weights = [AS209[k] for k in ['u', 'v', 'V', 'weights']]

    Rmax = 1.6

    FF = FrankFitter(Rmax, 20, geometry, alpha=1.05, weights_smooth=1e-2)

    sol = FF.fit(u, v, vis, weights)
    I_nn = sol.solve_non_negative()

    expected = np.array([
        2.42756717e+10, 1.28541672e+10, 1.90032938e+10, 8.31444339e+09,
        1.30814112e+10, 1.59442160e+09, 2.93990783e+08, 5.29739902e+09,
        1.24011568e+09, 5.40689479e+08, 1.97475180e+08, 2.12294162e+08,
        4.45700329e+09, 1.67658919e+09, 8.61662448e+08, 3.81032165e+08,
        2.41202443e+08, 7.68452028e+07, 0.00000000e+00, 2.86208170e+07
    ])

    np.testing.assert_allclose(I_nn, expected,
                               err_msg="Testing Frank Fit to AS 209")


def test_frank_fitter_log_normal():
    """Check FrankFitter fitting routine with AS 209 dataset"""
    AS209, geometry = load_AS209(uv_cut=1e6)

    u, v, vis, weights = [AS209[k] for k in ['u', 'v', 'V', 'weights']]
    
    Rmax = 1.6

    FF = FrankFitter(Rmax, 20, geometry, alpha=1.3, weights_smooth=1e-2,
                     method='LogNormal')

    sol = FF.fit(u, v, vis, weights)
    expected = np.array([
        2.36087004e+10, 1.36923798e+10, 1.82805612e+10, 8.72975548e+09,
        1.30516037e+10, 1.28158462e+09, 8.14949172e+08, 4.74472433e+09,
        1.66592277e+09, 3.39438704e+08, 1.56219080e+08, 4.42345087e+08,
        4.13155298e+09, 2.00246824e+09, 6.07773834e+08, 5.34020982e+08,
        1.80820913e+08, 7.71858927e+07, 2.89354816e+07, 2.45967370e+06,])

    np.testing.assert_allclose(sol.MAP, expected, rtol=7e-5,
                               err_msg="Testing Frank Log-Normal Fit to AS 209")


def test_geom_deproject():
    """Check predict works properly with a different geometry"""
    AS209, geometry = load_AS209(uv_cut=1e6)

    u, v, vis, weights = [AS209[k] for k in ['u', 'v', 'V', 'weights']]

    Rmax = 1.6

    FF = FrankFitter(Rmax, 20, geometry, alpha=1.05, weights_smooth=1e-2)
    sol = FF.fit(u, v, vis, weights)

    geom2 = FixedGeometry(0,0,0,0)

    q = np.hypot(u,v)
    q = np.geomspace(q.min(), q.max(), 20)
    
    V = sol.predict(q, 0*q, geometry=geom2)
    Vexpected = [ 
        0.3152656,   0.31260669,  0.30831366,  0.30143546,
        0.29055445,  0.27369896,  0.24848676,  0.2129406,
        0.1677038,   0.11989993,  0.08507635,  0.07491307,
        0.06555019,  0.01831576, -0.00173855,  0.00042803,
       -0.00322264,  0.00278782, -0.00978981,  0.00620259,
    ]
    
    np.testing.assert_allclose(V, Vexpected, rtol=2e-5, atol=1e-8,
                               err_msg="Testing predict with different geometry")


def test_fit_geometry_inside():
    """Check the geometry fit embedded in a call to FrankFitter"""
    AS209, _ = load_AS209(uv_cut=1e6)

    u, v, vis, weights = [AS209[k][::100] for k in ['u', 'v', 'V', 'weights']]

    Rmax = 1.6

    FF = FrankFitter(Rmax, 20, FitGeometryGaussian(),
                     alpha=1.05, weights_smooth=1e-2)

    sol = FF.fit(u, v, vis, weights)

    geom = sol.geometry
    np.testing.assert_allclose([geom.inc, geom.PA, 1e3 * geom.dRA,
                                1e3 * geom.dDec],
                               [34.50710460482996, 86.4699107557648,
                                0.21017246809441995, -2.109586872914908],
                               err_msg="Gaussian geometry fit inside Frank fit")


def test_throw_error_on_bad_q_range():
    """Check that frank correctly raises an error when the
    q range is bad."""
    AS209, geometry = load_AS209()

    u, v, vis, weights = [AS209[k][::100] for k in ['u', 'v', 'V', 'weights']]

    Rmax = 1.6

    FF = FrankFitter(Rmax, 20, geometry,
                     alpha=1.05, weights_smooth=1e-2)

    try:
        FF.fit(u, v, vis, weights)
        raise RuntimeError("Expected ValueError due to bad range")
    except ValueError:
        pass


def test_uvbin():
    """Check the uv-data binning routine"""
    AS209, geometry = load_AS209()

    uv = np.hypot(*geometry.deproject(AS209['u'], AS209['v']))

    uvbin = utilities.UVDataBinner(uv, AS209['V'], AS209['weights'], 50e3)

    uvmin = 1e6
    uvmax = 1e6 + 50e3

    idx = (uv >= uvmin) & (uv < uvmax)

    widx = AS209['weights'][idx]

    w = np.sum(widx)
    V = np.sum(widx*AS209['V'][idx]) / w
    q = np.sum(widx*uv[idx]) / w

    i = (uvbin.uv >= uvmin) & (uvbin.uv < uvmax)

    np.testing.assert_allclose(q, uvbin.uv[i])
    np.testing.assert_allclose(V, uvbin.V[i])
    np.testing.assert_allclose(w, uvbin.weights[i])
    np.testing.assert_allclose(len(widx), uvbin.bin_counts[i])

def test_save_load_sol():
    """Check saving/loading a frank 'sol' object"""
    AS209, AS209_geometry = load_AS209(uv_cut=1e6)
    u, v, vis, weights = [AS209[k][::100] for k in ['u', 'v', 'V', 'weights']]
    Rmax, N = 1.6, 20

    # generate a sol from a standard frank fit
    FF = FrankFitter(Rmax, N, AS209_geometry, alpha=1.05, weights_smooth=1e-2)
    sol = FF.fit(u, v, vis, weights)

    # and from a frank debris fit (has additional keys over a standard fit sol)
    FF_deb = FrankDebrisFitter(Rmax, N, AS209_geometry, lambda x : 0.05 * x, 
                                alpha=1.05, weights_smooth=1e-2)
    sol_deb = FF_deb.fit(u, v, vis, weights)

    tmp_dir = '/tmp/frank/tests'
    os.makedirs(tmp_dir, exist_ok=True)

    save_prefix = [os.path.join(tmp_dir, 'standard'), os.path.join(tmp_dir, 'debris')]
    sols = [sol, sol_deb]

    for ii, jj in enumerate(save_prefix):
        # save the 'sol' object 
        save_fit(u, v, vis, weights, sols[ii], prefix=jj,
            save_profile_fit=False, save_vis_fit=False, save_uvtables=False
            )
        # load it
        load_sol(jj + '_frank_sol.obj')


def test_arcsec_baseline():
    """Check utilities.arcsec_baseline"""
    result = utilities.arcsec_baseline(1e6)
    np.testing.assert_almost_equal(result, 0.2062648)


def test_radius_convert():
    """Check utilities.radius_convert"""
    result = utilities.radius_convert(2.0, 100)
    assert result == 200

    result_bwd = utilities.radius_convert(200.0, 100, conversion='au_arcsec')
    assert result_bwd == 2


def test_jy_convert():
    """Check utilities.jy_convert"""
    x = 10 

    bmaj, bmin = 0.1, 0.1

    expected = {'beam_sterad': 37547916727553.23, 
                'beam_arcsec2': 882.5424006, 
                'arcsec2_beam': 0.113309, 
                'arcsec2_sterad': 425451702961.5221, 
                'sterad_beam': 2.6632636e-12, 
                'sterad_arcsec2': 2.3504431e-10}
    
    for key in expected:
        result = utilities.jy_convert(x, conversion=key, bmaj=bmaj, bmin=bmin)
        np.testing.assert_almost_equal(result, expected[key], decimal=7)


def test_get_fit_stat_uncer():
    """Check utilities.get_fit_stat_uncer"""
    AS209, AS209_geometry = load_AS209(uv_cut=1e6)
    u, v, vis, weights = [AS209[k][::100] for k in ['u', 'v', 'V', 'weights']]

    # generate a sol from a standard frank fit
    FF = FrankFitter(1.6, 20, AS209_geometry, alpha=1.3, weights_smooth=1e-2)
    sol = FF.fit(u, v, vis, weights)    

    # call with normal model
    result = utilities.get_fit_stat_uncer(sol)
    expected = [4.29701157e+08, 4.38007127e+08, 3.81819050e+08, 2.80884179e+08,
       2.01486438e+08, 1.99436182e+08, 2.15565926e+08, 1.99285093e+08,
       1.65080363e+08, 1.49458838e+08, 1.55552558e+08, 1.55906224e+08,
       1.39929834e+08, 1.23399577e+08, 1.18687679e+08, 1.20329729e+08,
       1.19973246e+08, 1.15672282e+08, 1.05924406e+08, 7.19982652e+07]
    
    np.testing.assert_allclose(result, expected, rtol=2e-5, atol=1e-8)

    # call with lognormal model
    FF_logn = FrankFitter(1.6, 20, AS209_geometry, alpha=1.3, weights_smooth=1e-2,
                    method='LogNormal')
    sol_logn = FF.fit(u, v, vis, weights)    
    result_logn = utilities.get_fit_stat_uncer(sol_logn)    
    np.testing.assert_allclose(result_logn, expected, rtol=2e-5, atol=1e-8)


def test_normalize_uv():
    """Check utilities.normalize_uv"""
    result = utilities.normalize_uv(1e5, 1e5, 1e-3)
    np.testing.assert_almost_equal(result, (([1e8]), ([1e8])))


def test_cut_data_by_baseline():
    """Check utilities.cut_data_by_baseline"""
    AS209, AS209_geometry = load_AS209(uv_cut=1e6)
    u, v, vis, weights = [AS209[k][::100] for k in ['u', 'v', 'V', 'weights']]    

    # restrictive cut range to keep only a single baseline
    cut_range = [0, 11570]
    # call with no supplied geometry
    result = utilities.cut_data_by_baseline(u, v, vis, weights, cut_range)

    np.testing.assert_almost_equal(result, (([9105.87121309]),
                                            ([-7126.8802574]),
                                            ([0.25705367-0.00452954j]),
                                            ([14390.94693293])
                                            ))    

    # restrictive cut range to keep only a single baseline
    cut_range_geom = [0, 10370]
    # call with supplied geometry
    result_geom = utilities.cut_data_by_baseline(u, v, vis, weights, cut_range_geom, 
                                                 geometry=AS209_geometry)

    np.testing.assert_almost_equal(result_geom, (([3080.37968279]),
                                            ([-12126.45120077]),
                                            ([0.01581838-0.22529848j]),
                                            ([47.91090538])
                                            ))    
    

def test_estimate_weights():
    """Check utilities.estimate_weights"""
    AS209, AS209_geometry = load_AS209(uv_cut=1e6)
    u, v, vis, weights = [AS209[k][::100] for k in ['u', 'v', 'V', 'weights']]  
    
    # call with u, v, vis
    result = utilities.estimate_weights(u, v, vis)
    expected = [343.4011447938792, 1040.388154761485, 323.33779140104497, 
                670.5799827294733, 746.6778204045879, 262.537321708902, 
                916.0141170546902, 2478.5780126781183, 220.49922955106743,
                343.4011447938792]
    np.testing.assert_allclose(result[:10], expected, rtol=2e-5, atol=1e-8)

    # call with only u, vis
    result_no_v = utilities.estimate_weights(u, vis)
    expected_no_v = [140.15619775524289, 140.15619775524289,
                   136.20331899175486, 144.80828130035127,
                   751.9714145412686, 14.69762047498323,
                   775.4926337220135, 106.4511685363733,
                   188.5850930080213, 299.3538060369927]
    np.testing.assert_allclose(result_no_v[:10], expected_no_v, rtol=2e-5, atol=1e-8)

    # call with u, v, vis, use_median
    result_med = utilities.estimate_weights(u, v, vis, use_median=True)
    np.testing.assert_almost_equal(result_med[0], 1040.3881547614856)


def test_make_image():
    """Check utilities.make_image"""
    AS209, AS209_geometry = load_AS209(uv_cut=1e6)
    u, v, vis, weights = [AS209[k][::100] for k in ['u', 'v', 'V', 'weights']]

    # generate a sol from a standard frank fit
    FF = FrankFitter(1.6, 20, AS209_geometry, alpha=1.3, weights_smooth=1e-2)
    sol = FF.fit(u, v, vis, weights)   

    # call without projection
    result = utilities.make_image(sol, Npix=4, project=False)
    expected = (([-2.4, -0.8,  0.8,  2.4]),
                ([-2.4, -0.8,  0.8,  2.4]),
                ([[ 6.47513532e+06, -1.51846712e+07, -1.28084898e+08, -1.51846712e+07],
                  [-1.51846712e+07,  3.53545721e+07,  3.13428201e+08, 3.53545721e+07],
                  [-1.28084898e+08,  3.13428201e+08,  3.30602278e+09, 3.13428201e+08],
                  [-1.51846712e+07,  3.53545721e+07,  3.13428201e+08, 3.53545721e+07]])
                  )
    
    # check pixel coordinates
    np.testing.assert_allclose(result[:2], expected[:2], rtol=2e-5, atol=1e-8)
    # check pixel brightness
    Iresult = np.asarray(result[2])
    Iexpected = np.asarray(expected[2])
    np.testing.assert_allclose(Iresult, Iexpected, rtol=2e-5, atol=1e-8)

    # call with projection
    result_proj = utilities.make_image(sol, Npix=4, project=True)
    expected_proj = (([-2.4, -0.8,  0.8,  2.4]), 
                ([-2.4, -0.8,  0.8,  2.4]),
                ([[ 2.40226316e+06, -8.52280178e+06, -1.37522143e+08, -8.18630605e+06],
                  [-8.84143838e+06,  2.75149090e+07,  3.28577699e+08, 1.74953407e+07],
                  [-1.02387759e+08,  2.33877598e+08,  3.44906494e+09, 2.25119693e+08],
                  [-9.02378853e+06,  1.87158156e+07,  3.35326195e+08, 2.71103094e+07]])
                  )
    
    # check pixel coordinates
    np.testing.assert_allclose(result_proj[:2], expected_proj[:2], rtol=2e-5, atol=1e-8)
    # check pixel brightness
    Iresult_proj = np.asarray(result_proj[2])
    Iexpected_proj = np.asarray(expected_proj[2])
    np.testing.assert_allclose(Iresult_proj, Iexpected_proj, rtol=2e-5, atol=1e-8)


def test_add_vis_noise():
    """Check utilities.add_vis_noise"""
    # dummy vis and weight
    vis, weights = [1.1 + 0.9j], 0.5

    result = utilities.add_vis_noise(vis, weights, seed=47)
    np.testing.assert_almost_equal(result, [-0.0992665+2.74683048j])


def test_make_mock_data():
    """Check utilities.add_vis_noise"""
    AS209, AS209_geometry = load_AS209(uv_cut=1e6)
    u, v, vis, weights = [AS209[k][::100] for k in ['u', 'v', 'V', 'weights']]
    
    # generate a sol from a standard frank fit
    FF = FrankFitter(1.6, 20, AS209_geometry, alpha=1.3, weights_smooth=1e-2)
    sol = FF.fit(u, v, vis, weights)   

    # call with minimal inputs
    result = utilities.make_mock_data(sol.r, sol.I, 3.0, u, v)
    expected = [ 0.06699455,  0.18302498,  0.27758414,  0.04789997, -0.00240399,
        0.06339718,  0.00358722,  0.28862088,  0.07058801,  0.06617371]
    np.testing.assert_allclose(result[1][:10], expected, rtol=2e-5, atol=1e-8)

    # call with deprojection
    result_dep = utilities.make_mock_data(sol.r, sol.I, 3.0, u, v, 
                                      projection='deproject', 
                                      geometry=AS209_geometry)
    expected_dep = [0.06244746, 0.15925137, 0.2345302 , 0.0623711 , 0.00404342,
       0.06277, 0.00361453, 0.23649558, 0.06326574, 0.0632122 ]
    np.testing.assert_allclose(result_dep[1][:10], expected_dep, rtol=2e-5, atol=1e-8)

    # call with reprojection
    result_rep = utilities.make_mock_data(sol.r, sol.I, 3.0, u, v, 
                                      projection='reproject', 
                                      geometry=AS209_geometry)
    expected_rep = [ 0.05219592,  0.11866411,  0.22040989,  0.03889961, -0.00133919,
        0.05194375, -0.00054036,  0.23372006,  0.04987515,  0.04541111]
    np.testing.assert_allclose(result_rep[1][:10], expected_rep, rtol=2e-5, atol=1e-8)

    # call with added noise
    result_noi = utilities.make_mock_data(sol.r, sol.I, 3.0, u, v, 
                                          add_noise=True, weights=weights, seed=47)
    expected_noi = [-0.06817425,  0.3195001 ,  0.36992457,  0.11576222, -0.18251663,
        0.38046765, -0.13962233,  0.42048773,  0.01093563, -0.08652271]
    np.testing.assert_allclose(result_noi[1][:10], expected_noi, rtol=2e-5, atol=1e-8)


def test_get_collocation_points():
    """Check utilities.get_collocation_points"""
    # call with forward DHT
    result = utilities.get_collocation_points(N=10)
    expected = [0.14239924, 0.32686567, 0.51242148, 0.69822343, 0.88411873,
       1.07005922, 1.25602496, 1.44200623, 1.62799772, 1.8139963 ]
    np.testing.assert_allclose(result, expected, rtol=2e-5, atol=1e-8)

    # call with backward DHT
    result_bwd = utilities.get_collocation_points(N=10, direction='backward')
    expected_bwd = [ 39472.88305737,  90606.73736504, 142042.56471889, 193546.62066389,
       245076.55732463, 296619.01772663, 348168.47711355, 399722.24089812,
       451278.83939289, 502837.4032234 ]
    np.testing.assert_allclose(result_bwd, expected_bwd, rtol=2e-5, atol=1e-8)    
    
def test_prob_model():
    """Check the probabilities for the frank model"""
    AS209, AS209_geometry = load_AS209(uv_cut=1e6)
    u, v, vis, weights = [AS209[k][::100] for k in ['u', 'v', 'V', 'weights']]
    
    # generate a sol from a standard frank fit
    FF = FrankFitter(1.6, 20, AS209_geometry, alpha=1.3, weights_smooth=1e-2)
    sol = FF.fit(u, v, vis, weights)  

    result = FF.log_evidence_laplace()
    np.testing.assert_allclose([result], [18590.205198687152], rtol=1e-7)


def _run_pipeline(geometry='gaussian', fit_phase_offset=True,
                   fit_inc_pa=True, make_figs=False,
                   multifit=False, bootstrap=False):
    """Check the full pipeline that performs a fit and outputs results"""

    # Build a subset of the data that we'll load during the fit
    AS209, AS209_geometry = load_AS209(uv_cut=1e6)

    u, v, vis, weights = [AS209[k][::100] for k in ['u', 'v', 'V', 'weights']]

    tmp_dir = '/tmp/frank/tests'
    os.makedirs(tmp_dir, exist_ok=True)

    uv_table = os.path.join(tmp_dir, 'small_uv.npz')
    save_uvtable(uv_table, u, v, vis, weights)

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
    geom['fit_inc_pa'] = fit_inc_pa
    geom['inc'] = AS209_geometry.inc
    geom['pa'] = AS209_geometry.PA
    geom['dra'] = AS209_geometry.dRA
    geom['ddec'] = AS209_geometry.dDec

    if make_figs:
        params['plotting']['quick_plot'] = True
        params['plotting']['full_plot'] = True
        params['plotting']['diag_plot'] = True
        params['plotting']['deprojec_plot'] = True
        params['input_output']['save_figures'] = False
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
    _run_pipeline('gaussian', fit_phase_offset=True)


def test_pipeline_no_phase():
    """Check the full fit pipeline when only fitting for the disc's inc, PA"""
    _run_pipeline('gaussian', fit_phase_offset=False)

def test_pipeline_no_inc_no_pa():
    """Check the full fit pipeline when only fitting for the disc's phase center"""
    _run_pipeline('gaussian', fit_phase_offset=True, fit_inc_pa=False)


def test_pipeline_known_geom():
    """Check the full fit pipeline when supplying a known disc geometry"""
    _run_pipeline('known')


def test_pipeline_figure_generation():
    """Check the full fit pipeline when producing all figures"""
    _run_pipeline('known', make_figs=True)


def test_pipeline_multifit():
    """Check the full fit pipeline when producing all figures and running multiple fits"""
    _run_pipeline('known', multifit=True)


def test_pipeline_bootstrap():
    """Check the full fit pipeline when producing all figures and running bootstrap"""
    _run_pipeline('known', bootstrap=True)
