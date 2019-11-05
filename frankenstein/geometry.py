"""This module contains methods for fitting the geometry and deprojecting the visibilties.
"""

from __future__ import division, absolute_import, print_function

import numpy as np
from scipy.optimize import minimize

__all__  = [ "apply_phase_shift", "deproject", "fit_geometry_gaussian",
             "SourceGeometry"]

def apply_phase_shift(u, v, vis,  dRA, dDec, inverse=False):    
    """Shift the phase centre of the visibilties.
    
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
    dRA *= 2. * np.pi 
    dDec *= 2. * np.pi

    phi = (u * dRA + v * dDec) / rad_2_arcsec

    return vis * (np.cos(phi) + 1j * np.sin(phi))

def deproject(u, v, inc, PA, inverse=False):
    """De-project the image in visibily space

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
    """Centres and deprojects the source to ensure axisymmetry.
    
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
        self._inc  = inc
        self._PA   = PA
        self._dRA  = dRA
        self._dDec = dDec
    
    def apply_correction(self, u, v, vis):
        """Correct the phase-centre and de-project the visibilities.
        
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
        vis = apply_phase_shift(u, v, vis,  self._dRA, self._dDec)
        u, v = deproject(u,v, self._inc, self._PA)

        return u, v, vis
        
    def undo_correction(self, u, v, vis):
        """Undo the phase-centre correction and de-projection.
        
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
        u, v = deproject(u,v, self._inc, self._PA, inverse=True)
        vis = apply_phase_shift(u, v, vis,  -self._dRA, -self._dDec)

        return u, v, vis
    
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
        
        
def fit_geometry_gaussian(u,v, visib, weights, phase_centre=None):
    """Esimate the source geometry by fitting a Gaussian in uv-space.
    
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
    
    def _chi2_gauss(params):
        """Evaluate the Chi^2 of Gaussian fit"""
        dRA, dDec, inc, pa, norm, scal = params

        if phase_centre is None:
            vis_corr = apply_phase_shift(u, v, vis, dRA, dDec)
        else:
            vis_corr = vis
            
        up, vp = deproject(u,v, inc, pa)

        #Evaluate the gaussian:
        gaus = np.exp(- 0.5 * (up**2 + vp**2) / (rad_2_arcsec*scal)**2)

        # Evaluate at the Chi2, using gaussian tapering:
        chi2 = weights * np.abs(norm * gaus - vis_corr)**2
        return chi2.sum() / (2 * len(weights))
    
    if phase_centre is not None:
        vis = apply_phase_shift(u, v, vis, phase_centre[0], phase_centre[1])
    else:
        vis = visib
    
    res = minimize(_chi2_gauss, [0.0, 0.0,
                                 0.1, 0.1, 
                                 1.0, 1.0])

    dRA, dDec, inc, PA, _, _ = res.x
    
    if phase_centre is not None:
        dRA, dDec = phase_centre
    
    return SourceGeometry(inc, PA, dRA, dDec)
    
    
