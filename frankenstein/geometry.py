"""This module contains methods for fitting the geometry and deprojecting the visibilties.
"""

from __future__ import division, absolute_import, print_function

import numpy as np
from scipy.optimize import least_squares

from frankenstein.constants import rad_to_arcsec

__all__ = ["apply_phase_shift", "deproject", "fit_geometry_gaussian",
           "SourceGeometry"]


def apply_phase_shift(u, v, vis, dRA, dDec, inverse=False):
    """
    Shift the phase centre of the visibilties.

    Corrects the image centering in visibility space

    Parameters
    ----------
    u : array of real, size=N
        u-points of the visibilities
    v : array of real, size=N
        v-points of the visibilities
    vis : array of real, size=N
        Complex visibilites
    dRA : float, unit=arcseconds
        Phase-shift in Right Ascenion
    dDec : float, unit=arcseconds
        Phase-shift in Declination

    Returns
    -------
    shifted_vis : array of real, size=N
        Phase shifted visibilites.
    
    """
    dRA *= 2. * np.pi / rad_to_arcsec
    dDec *= 2. * np.pi / rad_to_arcsec

    phi = u * dRA + v * dDec

    return vis * (np.cos(phi) + 1j * np.sin(phi))


def deproject(u, v, inc, PA, inverse=False):
    """
    De-project the image in visibily space

    Parameters
    ----------
    u : array of real, size=N
        u-points of the visibilities
    v : array of real, size=N
        v-points of the visibilities
    vis : array of real, size=N
        Complex visibilites
    inc : float, unit=radians
        Inclination
    PA : float, unit=radians
        Position Angle
    inverse : bool, default=False
        If True the uv-points are re-projected rather than de-projected.

    Returns
    -------
    up : array, size=N
        Deprojected u-points
    vp : array, size=N
        Deprojected v-points
    
    """
    cos_t = np.cos(PA)
    sin_t = np.sin(PA)

    if inverse:
        sin_t *= -1

    up = u * cos_t - v * sin_t
    vp = u * sin_t + v * cos_t

    #   De-project
    if inverse:
        up /= np.cos(inc)
    else:
        up *= np.cos(inc)

    return up, vp


class SourceGeometry(object):
    """
    Centres and deprojects the source to ensure axisymmetry.

    Parameters
    ----------
    inc : float, unit=radians
        Inclination of the disc
    PA : float, unit=radians
        Position Angle of the disc
    dRA : float, unit=arcseconds
        Phase centre offset in Right Ascension
    dDec : float, unit=arcseconds
        Phase centre offset in Declination
    
    """

    def __init__(self, inc=0, PA=0, dRA=0, dDec=0):
        self._inc = inc
        self._PA = PA
        self._dRA = dRA
        self._dDec = dDec

    def apply_correction(self, u, v, vis):
        """
        Correct the phase-centre and de-project the visibilities.

        Parameters
        ----------
        u : array of real, size=N
            u-points of the visibilities
        v : array of real, size=N
            v-points of the visibilities
        vis : array of real, size=N
            Complex visibilites

        Returns
        -------
        up : array of real, size=N
            Corrected u-points of the visibilities
        vp : array of real, size=N
            Corrected v-points of the visibilities
        visp : array of real, size=N
            Corrected complex visibilites

        """
        vis = apply_phase_shift(u, v, vis, self._dRA, self._dDec)
        u, v = deproject(u, v, self._inc, self._PA)

        return u, v, vis

    def undo_correction(self, u, v, vis):
        """
        Undo the phase-centre correction and de-projection.

        Parameters
        ----------
        u : array of real, size=N
            u-points of the visibilities
        v : array of real, size=N
            v-points of the visibilities
        vis : array of real, size=N
            Complex visibilites

        Returns
        -------
        up : array of real, size=N
            Corrected u-points of the visibilities
        vp : array of real, size=N
            Corrected v-points of the visibilities
        visp : array of real, size=N
            Corrected complex visibilites

        """
        u, v = self.reproject(u, v)
        vis = apply_phase_shift(u, v, vis, -self._dRA, -self._dDec)

        return u, v, vis

    def deproject(self, u, v):
        """Convert uv-points to sky-plane deprojected space"""
        return deproject(u, v, self._inc, self._PA)

    def reproject(self, u, v):
        """Convert uv-points from deprojected space to sky-plane"""
        return deproject(u, v, self._inc, self._PA, inverse=True)

    @property
    def dRA(self):
        """Phase centre offset in Right Ascension"""
        return self._dRA

    @property
    def dDec(self):
        """Phase centre offset in Declination"""
        return self._dDec

    @property
    def PA(self):
        """Position angle of the disc"""
        return self._PA

    @property
    def inc(self):
        """Inclination of the disc"""
        return self._inc


def fit_geometry_gaussian(u, v, visib, weights, phase_centre=None):
    """
    Esimate the source geometry by fitting a Gaussian in uv-space.

    Parameters
    ----------
    u : array of real, size=N
        u-points of the visibilities
    v : array of real, size=N
        v-points of the visibilities
    vis : array of real, size=N
        Complex visibilites
    phase_centre: [dRA, dDec], optional. 
        The Phase centre offsets dRA and dDec in arcseconds.
        If not provided, these will be fit for.

    Returns
    -------
    geometry : SourceGeometry object
        Fitted geometry.
    
    """
    fac = 2*np.pi / rad_to_arcsec
    w = np.sqrt(weights)

    def wrap(fun):
        return np.concatenate([fun.real, fun.imag])
        
    def _gauss_fun(params):
        dRA, dDec, inc, pa, norm, scal = params

        if phase_centre is None:
            phi = dRA*fac * u + dDec*fac * v
            Vp = visib * (np.cos(phi) + 1j*np.sin(phi))
        else:
            Vp = visib

        c_t = np.cos(pa)
        s_t = np.sin(pa)
        c_i = np.cos(inc)
        up = (u*c_t - v*s_t) * c_i / (scal*rad_to_arcsec)
        vp = (u*s_t + v*c_t)  / (scal*rad_to_arcsec)

        fun = w*(norm * np.exp(-0.5*(up*up+ vp*vp)) - Vp)

        return wrap(fun)

    def _gauss_jac(params):
        dRA, dDec, inc, pa, norm, scal = params

        jac = np.zeros([6, 2*len(w)])

        if phase_centre is None:
            phi = dRA*fac * u + dDec*fac * v
            dVp = - w*visib * (-np.sin(phi) + 1j*np.cos(phi)) * fac

            jac[0] = wrap(dVp*u)
            jac[1] = wrap(dVp*v)

        c_t = np.cos(pa)
        s_t = np.sin(pa)
        c_i = np.cos(inc)
        s_i = np.sin(inc)
        up = (u*c_t - v*s_t) 
        vp = (u*s_t + v*c_t) 

        uv = (up*up*c_i*c_i+ vp*vp)

        G = w*np.exp(-0.5 * uv / (scal*rad_to_arcsec)**2)

        norm = norm / (scal*rad_to_arcsec)**2

        jac[2] = wrap(norm*G*up*up*c_i*s_i)
        jac[3] = wrap(norm*G*up*vp*(c_i*c_i - 1)/2)
        jac[4] = wrap(G)
        jac[5] = wrap(norm*G*uv/scal)

        return jac.T


    res = least_squares(_gauss_fun,[0.0, 0.0,
                                    0.1, 0.1,
                                    1.0, 1.0],
                        jac=_gauss_jac, method='lm')

    dRA, dDec, inc, PA, _, _ = res.x

    if phase_centre is not None:
        dRA, dDec = phase_centre

    return SourceGeometry(inc, PA, dRA, dDec)
