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
"""This module contains methods for fitting a source's geometry and deprojecting
   the visibilties by a fitted or a given geometry.

   NOTE: The sign convention used here is xx.
"""

import numpy as np
from scipy.optimize import least_squares

from frank.constants import rad_to_arcsec, deg_to_rad

__all__ = ["SourceGeometry", "FixedGeometry", "FitGeometryGaussian"]


def apply_phase_shift(u, v, V, dRA, dDec, inverse=False):
    r"""
    Shift the phase centre of the visibilties.

    Correct the image centering in visibility space

    Parameters
    ----------
    u : array of real, size = N, unit = :math:`\lambda`
        u-points of the visibilities
    v : array of real, size = N, unit = :math:`\lambda`
        v-points of the visibilities
    V : array of real, size = N, unit = Jy
        Complex visibilites
    dRA : float, unit = arcsec
        Phase shift in right ascenion.
        NOTE: The sign convention is xx
    dDec : float, unit = arcsec
        Phase shift in declination.
        NOTE: The sign convention is xx

    Returns
    -------
    shifted_vis : array of real, size = N, unit = Jy
        Phase shifted visibilites

    """
    dRA *= 2. * np.pi / rad_to_arcsec
    dDec *= 2. * np.pi / rad_to_arcsec

    phi = u * dRA + v * dDec

    return V * (np.cos(phi) + 1j * np.sin(phi))


def deproject(u, v, inc, PA, inverse=False):
    r"""
    Deproject the image in visibily space

    Parameters
    ----------
    u : array of real, size = N, unit = :math:`\lambda`
        u-points of the visibilities
    v : array of real, size = N, unit = :math:`\lambda`
        v-points of the visibilities
    inc : float, unit = deg
        Inclination
    PA : float, unit = deg
        Position angle
    inverse : bool, default=False
        If True, the uv-points are reprojected rather than deprojected

    Returns
    -------
    up : array, size = N, unit = :math:`\lambda`
        Deprojected u-points
    vp : array, size = N, unit = :math:`\lambda`
        Deprojected v-points

    """

    inc *= deg_to_rad
    PA *= deg_to_rad

    cos_t = np.cos(PA)
    sin_t = np.sin(PA)

    if inverse:
        sin_t *= -1

    up = u * cos_t - v * sin_t
    vp = u * sin_t + v * cos_t

    # Deproject
    if inverse:
        up /= np.cos(inc)
    else:
        up *= np.cos(inc)

    return up, vp


class SourceGeometry(object):
    """
    Base class for geometry corrections.

    Centre and deproject the source to ensure axisymmetry

    Parameters
    ----------
    inc : float, unit = deg
        Inclination of the disc
    PA : float, unit = deg
        Position angle of the disc
    dRA : float, unit = arcsec
        Phase centre offset in right ascension.
        NOTE: The sign convention is xx
    dDec : float, units = arcsec
        Phase centre offset in declination.
        NOTE: The sign convention is xx

    """

    def __init__(self, inc=None, PA=None, dRA=None, dDec=None):
        self._inc = inc
        self._PA = PA
        self._dRA = dRA
        self._dDec = dDec

    def apply_correction(self, u, v, V):
        r"""
        Correct the phase centre and deproject the visibilities

        Parameters
        ----------
        u : array of real, size = N, unit = :math:`\lambda`
            u-points of the visibilities
        v : array of real, size = N, unit = :math:`\lambda`
            v-points of the visibilities
        V : array of real, size = N, units = Jy
            Complex visibilites

        Returns
        -------
        up : array of real, size = N, unit = :math:`\lambda`
            Corrected u-points of the visibilities
        vp : array of real, size = N, unit = :math:`\lambda`
            Corrected v-points of the visibilities
        Vp : array of real, size = N, unit = Jy
            Corrected complex visibilites

        """
        V = apply_phase_shift(u, v, V, self._dRA, self._dDec)
        u, v = deproject(u, v, self._inc, self._PA)

        return u, v, V

    def undo_correction(self, u, v, V):
        r"""
        Undo the phase centre correction and deprojection

        Parameters
        ----------
        u : array of real, size = N, unit = :math:`\lambda`
            u-points of the visibilities
        v : array of real, size = N, unit = :math:`\lambda`
            v-points of the visibilities
        V : array of real, size = N, unit = Jy
            Complex visibilites

        Returns
        -------
        up : array of real, size = N, unit = :math:`\lambda`
            Corrected u-points of the visibilities
        vp : array of real, size = N, unit = :math:`\lambda`
            Corrected v-points of the visibilities
        Vp : array of real, size = N, unit = Jy
            Corrected complex visibilites
        """

        u, v = self.reproject(u, v)
        vis = apply_phase_shift(u, v, V, -self._dRA, -self._dDec)

        return u, v, V

    def deproject(self, u, v):
        """Convert uv-points from sky-plane to deprojected space"""
        return deproject(u, v, self._inc, self._PA)

    def reproject(self, u, v):
        """Convert uv-points from deprojected space to sky-plane"""
        return deproject(u, v, self._inc, self._PA, inverse=True)

    def fit(self, u, v, V, weights):
        r"""
        Determine geometry using the provided uv-data

        Parameters
        ----------
        u : array of real, size = N, unit = :math:`\lambda`
            u-points of the visibilities
        v : array of real, size = N, unit = :math:`\lambda`
            v-points of the visibilities
        V : array of complex, size = N, unit = Jy
            Complex visibilites
        weights : array of real, size = N, unit = Jy
            Weights on the visibilities
        """
        return

    def clone(self):
        """Save the geometry parameters in a seperate geometry object"""
        return FixedGeometry(self.inc, self.PA, self.dRA, self.dDec)

    @property
    def dRA(self):
        """Phase centre offset in right ascension, unit = arcsec"""
        return self._dRA

    @property
    def dDec(self):
        """Phase centre offset in declination, unit = arcsec"""
        return self._dDec

    @property
    def PA(self):
        """Position angle of the disc, unit = rad"""
        return self._PA

    @property
    def inc(self):
        """Inclination of the disc, unit = rad"""
        return self._inc


class FixedGeometry(SourceGeometry):
    """
    Disc Geometry class using pre-determined parameters.

    Centre and deproject the source to ensure axisymmetry

    Parameters
    ----------
    inc : float, unit = deg
        Disc inclination
    PA : float, unit = deg
        Disc positition angle.
    dRA : float, default = 0, unit = arcsec
        Phase centre offset in right ascension
        NOTE: The sign convention is xx
    dDec : float, default = 0, unit = arcsec
        Phase centre offset in declination
        NOTE: The sign convention is xx

    """

    def __init__(self, inc, PA, dRA=0.0, dDec=0.0):
        super(FixedGeometry, self).__init__(inc, PA, dRA, dDec)


class FitGeometryGaussian(SourceGeometry):
    """
    Determine the disc geometry by fitting a Gaussian in Fourier space.

    Centre and deproject the source to ensure axisymmetry

    Parameters
    ----------
    phase_centre : tuple = (dRA, dDec) or None (default), unit = arcsec
         Determine whether to fit for the source's phase centre. If
         phase_centre = None, the phase centre is fit for. Else the phase
         centre should be provided as a tuple

    """

    def __init__(self, phase_centre=None):
        super(FitGeometryGaussian, self).__init__()

        self._phase_centre = phase_centre

    def fit(self, u, v, V, weights):
        r"""
        Determine geometry using the provided uv-data

        Parameters
        ----------
        u : array of real, size = N, unit = :math:`\lambda`
            u-points of the visibilities
        v : array of real, size = N, unit = :math:`\lambda`
            v-points of the visibilities
        V : array of complex, size = N, unit = Jy
            Complex visibilites
        weights : array of real, size=N, unit = Jy^-2
            Weights on the visibilities
        """

        inc, PA, dRA, dDec = _fit_geometry_gaussian(
            u, v, V, weights, phase_centre=self._phase_centre)

        self._inc = inc
        self._PA = PA
        self._dRA = dRA
        self._dDec = dDec


def _fit_geometry_gaussian(u, v, V, weights, phase_centre=None):
    r"""
    Estimate the source geometry by fitting a Gaussian in uv-space

    Parameters
    ----------
    u : array of real, size = N, unit = :math:`\lambda`
        u-points of the visibilities
    v : array of real, size = N, unit = :math:`\lambda`
        v-points of the visibilities
    V : array of complex, size = N, unit = Jy
        Complex visibilites
    weights : array of real, size = N, unit = Jy^-2
        Weights on the visibilities
    phase_centre: [dRA, dDec], optional, unit = arcsec
        The phase centre offsets dRA and dDec.
        If not provided, these will be fit for

    Returns
    -------
    geometry : SourceGeometry object
        Fitted geometry
    """
    fac = 2*np.pi / rad_to_arcsec
    w = np.sqrt(weights)

    def wrap(fun):
        return np.concatenate([fun.real, fun.imag])

    def _gauss_fun(params):
        dRA, dDec, inc, pa, norm, scal = params

        if phase_centre is None:
            phi = dRA*fac * u + dDec*fac * v
            Vp = V * (np.cos(phi) + 1j*np.sin(phi))
        else:
            Vp = V

        c_t = np.cos(pa)
        s_t = np.sin(pa)
        c_i = np.cos(inc)
        up = (u*c_t - v*s_t) * c_i / (scal*rad_to_arcsec)
        vp = (u*s_t + v*c_t) / (scal*rad_to_arcsec)

        fun = w*(norm * np.exp(-0.5*(up*up + vp*vp)) - Vp)

        return wrap(fun)

    def _gauss_jac(params):
        dRA, dDec, inc, pa, norm, scal = params

        jac = np.zeros([6, 2*len(w)])

        if phase_centre is None:
            phi = dRA*fac * u + dDec*fac * v
            dVp = - w*V * (-np.sin(phi) + 1j*np.cos(phi)) * fac

            jac[0] = wrap(dVp*u)
            jac[1] = wrap(dVp*v)

        c_t = np.cos(pa)
        s_t = np.sin(pa)
        c_i = np.cos(inc)
        s_i = np.sin(inc)
        up = (u*c_t - v*s_t)
        vp = (u*s_t + v*c_t)

        uv = (up*up*c_i*c_i + vp*vp)

        G = w*np.exp(-0.5 * uv / (scal*rad_to_arcsec)**2)

        norm = norm / (scal*rad_to_arcsec)**2

        jac[2] = wrap(norm*G*up*up*c_i*s_i)
        jac[3] = wrap(norm*G*up*vp*(c_i*c_i - 1)/2)
        jac[4] = wrap(G)
        jac[5] = wrap(norm*G*uv/scal)

        return jac.T

    res = least_squares(_gauss_fun, [0.0, 0.0,
                                     0.1, 0.1,
                                     1.0, 1.0],
                        jac=_gauss_jac, method='lm')

    dRA, dDec, inc, PA, _, _ = res.x

    if phase_centre is not None:
        dRA, dDec = phase_centre

    return inc / deg_to_rad, PA / deg_to_rad, dRA, dDec
