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
"""This module contains functions for computing the
discrete Hankel transform (DHT).
"""

import numpy as np
from scipy.special import j0, j1, jn_zeros, jv

from frank.constants import rad_to_arcsec


class DiscreteHankelTransform(object):
    r"""
    Utilities for computing the discrete Hankel transform.

    This class provides the necessary interface to compute
    a discrete version of the Hankel transform (DHT):

        H[f](q) = \int_0^R_{max} f(r) J_nu(2*pi*q*r) * 2*pi*r dr.

    The DHT is based on [1].

    Additionally this class provides coefficients of the DHT [1] transform
    matrix

    Parameters
    ----------
    Rmax : float
        Maximum radius beyond which f(r) is zero
    N : integer
        Number of terms to use in the series
    nu : integer, default = 0
        Order of the Bessel function, J_nu(r)

    References
    ----------
    [1] Baddour & Chouinard (2015)
        DOI: https://doi.org/10.1364/JOSAA.32.000611
        Note: the definition of the DHT used here differs by factors
        of 2*pi.
    """
    def __init__(self, Rmax, N, nu=0):

        # Select the fast Bessel functions, if available
        if nu == 0:
            self._jnu0 = j0
            self._jnup = j1
        elif nu == 1:
            self._jnu0 = j1
            self._jnup = lambda x: jv(2, x)
        else:
            self._jnu0 = lambda x: jv(nu, x)
            self._jnup = lambda x: jv(nu + 1, x)

        self._N = N
        self._nu = nu

        # Compute the collocation points
        j_nk = jn_zeros(nu, N + 1)
        j_nk, j_nN = j_nk[:-1], j_nk[-1]

        Qmax = j_nN / (2 * np.pi * Rmax)

        self._Rnk = Rmax * (j_nk / j_nN)
        self._Qnk = Qmax * (j_nk / j_nN)

        self._Rmax = Rmax
        self._Qmax = Qmax

        # Compute the weights matrix
        Jnk = np.outer(np.ones_like(j_nk), self._jnup(j_nk))

        self._Ykm = (2 / (j_nN * Jnk * Jnk)) * \
            self._jnu0(np.prod(np.meshgrid(j_nk, j_nk/j_nN), axis=0))

        self._scale_factor = 1 / self._jnup(j_nk) ** 2

        # Store the extra data needed
        self._j_nk = j_nk
        self._j_nN = j_nN

    @classmethod
    def get_collocation_points(cls, Rmax, N, nu=0):
        """
        Compute the collocation points for a Hankel Transform.

        Parameters
        ----------
        Rmax : float
            Maximum radius beyond which f(r) is zero
        N : integer
            Number of terms to use in the series
        nu : integer, default = 0
            Order of the Bessel function, J_nu(r)

        Returns
        -------
        Rnk : array, shape=(N,) unit=radians
            Radial collocation points in
        Qnk : array, shape=(N,) unit=1/radians
            Frequency collocation points
        """

        j_nk = jn_zeros(nu, N + 1)
        j_nk, j_nN = j_nk[:-1], j_nk[-1]

        Qmax = j_nN / (2 * np.pi * Rmax)

        Rnk = Rmax * (j_nk / j_nN)
        Qnk = Qmax * (j_nk / j_nN)

        return Rnk, Qnk

    def transform(self, f, q=None, direction='forward'):
        """
        Compute the Hankel transform of an array

        Parameters
        ----------
        f : array, size = N
            Function to Hankel transform, evaluated at the collocation points:
                f[k] = f(r_k) or f[k] = f(q_k)
        q : array or None
            The frequency points at which to evaluate the Hankel
            transform. If not specified, the conjugate points of the
            DHT will be used. For the backwards transform, q should be
            the radius points
        direction : { 'forward', 'backward' }, optional
            Direction of the transform. If not supplied, the forward
            transform is used

        Returns
        -------
        H[f] : array, size = N or len(q) if supplied
            The Hankel transform of the array f

        """
        if q is None:
            Y = self._Ykm

            if direction == 'forward':
                norm = (2 * np.pi * self._Rmax ** 2) / self._j_nN
            elif direction == 'backward':
                norm = (2 * np.pi * self._Qmax ** 2) / self._j_nN
            else:
                raise AttributeError("direction must be one of {}"
                                     "".format(['forward', 'backward']))
        else:
            Y = self.coefficients(q, direction=direction)
            norm = 1.0

        return norm * np.dot(Y, f)

    def coefficients(self, q=None, direction='forward'):
        """
        Coefficients of the transform matrix, defined by
            H[f](q) = np.dot(Y, f)

        Parameters
        ----------
        q : array or None
            Frequency points at which to evaluate the transform. If q = None,
            the points of the DHT are used. If direction='backward', these
            points should instead be the radius points
        direction : { 'forward', 'backward' }, optional
            Direction of the transform. If not supplied, the forward transform
            is used

        Returns
        -------
        Y : array, size = (len(q), N)
            The transformation matrix
        """
        if direction == 'forward':
            norm = 1 / (np.pi * self._Qmax ** 2)
            k = 1. / self._Qmax
        elif direction == 'backward':
            norm = 1 / (np.pi * self._Rmax ** 2)
            k = 1. / self._Rmax
        else:
            raise AttributeError("direction must be one of {}"
                                 "".format(['forward', 'backward']))

        # For the DHT points we can use the cached Ykm points
        if q is None:
            return 0.5 * self._j_nN * norm * self._Ykm

        H = (norm * self._scale_factor) * \
            self._jnu0(np.outer(k * q, self._j_nk))

        return H

    @property
    def r(self):
        """Radius points"""
        return self._Rnk

    @property
    def Rmax(self):
        """Maximum radius"""
        return self._Rmax

    @property
    def q(self):
        """Frequency points"""
        return self._Qnk

    @property
    def Qmax(self):
        """Maximum frequency"""
        return self._Qmax

    @property
    def size(self):
        """Number of points used in the DHT"""
        return self._N

    @property
    def order(self):
        """Order of the Bessel function"""
        return self._nu
