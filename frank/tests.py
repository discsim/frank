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
""" Tests """

import numpy as np

from frank.hankel import DiscreteHankelTransform
from frank.radial_fitters import FourierBesselFitter, FrankFitter
from frank.geometry import FixedGeometry, FitGeometryGaussian
from frank.constants import deg_to_rad


def test_hankel_gauss():
    """Check the Hankel Transform"""

    def gauss_real_space(r):
        x = r
        return np.exp(-0.5 * x * x)

    def gauss_vis_space(q):
        qs = (2 * np.pi) * q
        return np.exp(-0.5 * qs * qs) * (2 * np.pi)

    DHT = DiscreteHankelTransform(5.0, 100)

    Ir = gauss_real_space(DHT.r)
    Iq = gauss_vis_space(DHT.q)

    # Test at the DHT points
    # Use a large error estimate because the DHT is approximate.
    np.testing.assert_allclose(Iq, DHT.transform(Ir, direction='forward'),
                               atol=1e-5, rtol=0, err_msg="Forward DHT")

    np.testing.assert_allclose(Ir, DHT.transform(Iq, direction='backward'),
                               atol=1e-5, rtol=0, err_msg="Inverse DHT")

    # Test at generic points:
    #   Larger Error needed
    q = np.linspace(0.0, 1.0, 25)
    np.testing.assert_allclose(gauss_vis_space(q),
                               DHT.transform(Ir, q=q, direction='forward'),
                               atol=1e-4, rtol=0, err_msg="Generic Forward DHT")

    r = np.linspace(0, 5.0, 25)
    np.testing.assert_allclose(gauss_real_space(r),
                               DHT.transform(Iq, q=r, direction='backward'),
                               atol=1e-4, rtol=0, err_msg="Generic Inverse DHT")

    # Finally check the coefficients matrix works
    Hf = DHT.coefficients(direction='forward')
    Hb = DHT.coefficients(direction='backward')

    np.testing.assert_allclose(np.dot(Hf, Ir),
                               DHT.transform(Ir, direction='forward'),
                               rtol=1e-7, err_msg="Forward DHT Coeffs")
    np.testing.assert_allclose(np.dot(Hb, Iq),
                               DHT.transform(Iq, direction='backward'),
                               rtol=1e-7, err_msg="Inverse DHT Coeffs")

    # Compare Cached vs non-cached DHT points:
    np.testing.assert_allclose(DHT.coefficients(q=DHT.q),
                               DHT.coefficients(),
                               atol=1e-12, rtol=0, err_msg="Cached forward DHT Coeffs")
    np.testing.assert_allclose(DHT.coefficients(q=DHT.r, direction='backward'),
                               DHT.coefficients(direction='backward'),
                               atol=1e-12, rtol=0, err_msg="Cached inverse DHT Coeffs")


def test_import_data():
    # TODO
    pass


def load_AS209():
    uv_AS209_DHSARP = np.load('tutorials/AS209_continuum.npz')
    geometry = FixedGeometry(dRA=1.9e-3, dDec=-2.5e-3, inc=34.97,
                              PA=85.76)

    return uv_AS209_DHSARP, geometry


def test_fit_geometry():
    # Check the geometry fit on a subset of the AS209 data
    AS209, _ = load_AS209()
    u, v, vis, weights = [AS209[k][::100] for k in ['u', 'v', 'V', 'weights']]

    geom = FitGeometryGaussian()
    geom.fit(u, v, vis, weights)

    np.testing.assert_allclose([geom.PA, geom.inc, 1e3 * geom.dRA, 1e3 * geom.dDec],
                               [1.4916013559412147 / deg_to_rad,
                               -0.5395904796783955 / deg_to_rad,
                                0.6431627790617276, 1.161768824369382],
                               err_msg="Gaussian geometry fit")


def test_fit_geometry_inside():
    # Check the geometry fit while the Frank fit is executed
    AS209, _ = load_AS209()

    u, v, vis, weights = [AS209[k][::100] for k in ['u', 'v', 'V', 'weights']]

    Rmax = 1.6

    FF = FrankFitter(Rmax, 20, FitGeometryGaussian(),
                     alpha=1.05, weights_smooth=1e-2)

    sol = FF.fit(u, v, vis, weights)

    geom = sol.geometry
    np.testing.assert_allclose([geom.PA, geom.inc, 1e3 * geom.dRA, 1e3 * geom.dDec],
                               [1.4916013559412147  / deg_to_rad,
                               -0.5395904796783955  / deg_to_rad,
                                0.6431627790617276, 1.161768824369382],
                               err_msg="Gaussian geometry fit inside Frank fit")


def test_fourier_bessel_fitter():
    """ Run Frank on AS 209 dataset in tutorials directory """

    AS209, geometry = load_AS209()

    u, v, vis, weights = [AS209[k] for k in ['u', 'v', 'V', 'weights']]

    Rmax = 1.6

    FB = FourierBesselFitter(Rmax, 20, geometry=geometry)

    sol = FB.fit(u, v, vis, weights)

    expected = np.array([1.89446696e+10, 1.81772972e+10, 1.39622125e+10, 1.20709653e+10,
                         9.83716859e+09, 3.26308106e+09, 2.02453146e+08, 4.73919867e+09,
                         1.67911877e+09, 1.73161931e+08, 4.50233539e+08, 3.57108466e+08,
                         4.04216831e+09, 1.89085113e+09, 6.73819228e+08, 5.50895976e+08,
                         1.53683576e+08, 1.02413038e+08, 2.32589333e+07, 3.33260713e+07])

    np.testing.assert_allclose(sol.mean, expected,
                               err_msg="Testing FourierBessel Fit to AS 209")


def test_frank_fitter():
    """ Run Frank on AS 209 dataset in tutorials directory """

    AS209, geometry = load_AS209()

    u, v, vis, weights = [AS209[k] for k in ['u', 'v', 'V', 'weights']]

    Rmax = 1.6

    FF = FrankFitter(Rmax, 20, geometry, alpha=1.05, weights_smooth=1e-2)

    sol = FF.fit(u, v, vis, weights)

    expected = np.array([1.89447670e+10, 1.81771688e+10, 1.39622994e+10, 1.20709223e+10,
                         9.83714570e+09, 3.26303606e+09, 2.02683225e+08, 4.73895907e+09,
                         1.67919735e+09, 1.73191351e+08, 4.50154739e+08, 3.57282237e+08,
                         4.04196935e+09, 1.89094928e+09, 6.73782223e+08, 5.50924438e+08,
                         1.53674025e+08, 1.02419407e+08, 2.32503410e+07, 3.33286294e+07])

    np.testing.assert_allclose(sol.mean, expected,
                               err_msg="Testing Frank Fit to AS 209")
