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
"""This module contains methods for fitting a radial brightness profile to a set
  of deprojected visibities.
"""
import abc
from collections import defaultdict
import logging
import numpy as np

from frank.constants import deg_to_rad, rad_to_arcsec
from frank.filter import CriticalFilter
from frank.hankel import DiscreteHankelTransform
from frank.statistical_models import (
    GaussianModel, LogNormalMAPModel, VisibilityMapping
)

class FrankRadialFit(metaclass=abc.ABCMeta):
    """
    Base class for results of frank fits.

    Parameters
    ----------
    vis_map : VisibilityMapping object
        Mapping between image and visibility plane. 
    info: dict
        Dictionary containing useful quantities for reproducing a fit
        (such as the hyperparameters used)
    geometry: SourceGeometry object, optional
        Geometry used to correct the visibilities for the source
        inclination. If not provided, the geometry determined during the
        fit will be used.
    """
    def __init__(self, vis_map, info, geometry):
        self._vis_map = vis_map
        self._geometry = geometry
        self._info = info

    def predict(self, u, v, I=None, geometry=None):
        r"""
        Predict the visibilities in the sky-plane

        Parameters
        ----------
        u, v : array, unit = :math:`\lambda`
            uv-points to predict the visibilities at
        I : array, optional, unit = Jy
            Intensity points to predict the vibilities of. If not specified,
            the mean will be used. The intensity should be specified at the
            collocation points, I[k] = :math:`I(r_k)`
        geometry: SourceGeometry object, optional
            Geometry used to correct the visibilities for the source
            inclination. If not provided, the geometry determined during the
            fit will be used

        Returns
        -------
        V(u,v) : array, unit = Jy
            Predicted visibilties of a source with a radial flux distribution
            given by :math:`I` and the position angle, inclination and phase
            centre determined by the geometry object
        """
        if geometry is None:
            geometry = self._geometry

        if I is None:
            I = self.I

        if geometry is not None:
            u, v, wz = geometry.deproject(u, v, use3D=True)
        else:
            wz = np.zeros_like(u)
            
        q = np.hypot(u, v)
        V = self._vis_map.predict_visibilities(I, q, wz, geometry=geometry)

        # Undo phase centering
        if geometry is not None:
            _, _, V = geometry.undo_correction(u, v, V)

        return V

    def predict_deprojected(self, q=None, I=None, geometry=None,
                            block_size=10**5, assume_optically_thick=True):
        r"""
        Predict the visibilities in the deprojected-plane

        Parameters
        ----------
        q : array, default = self.q, unit = :math:`\lambda`
            1D uv-points to predict the visibilities at
        I : array, optional, unit = Jy / sr
            Intensity points to predict the vibilities of. If not specified,
            the mean will be used. The intensity should be specified at the
            collocation points, I[k] = I(r_k)
        geometry: SourceGeometry object, optional
            Geometry used to correct the visibilities for the source
            inclination. If not provided, the geometry determined during the
            fit will be used
        block_size : int, default = 10**5
            Maximum matrix size used in the visibility calculation
        assume_optically_thick : bool, default = True
            Whether to correct the visibility amplitudes for the source
            inclination

        Returns
        -------
        V(q) : array, unit = Jy
            Predicted visibilties of a source with a radial flux distribution
            given by :math:`I`. The amplitude of the visibilities are reduced
            according to the inclination of the source, for consistency with
            `uvplot`

        Notes
        -----
        The visibility amplitudes are still reduced due to the projection,
        for consistency with `uvplot`
        """
        if geometry is None:
            geometry = self._geometry
       
        if I is None:
            I = self.I

        V = self._vis_map.predict_visibilities(I, q, q*0, geometry=geometry)

        return V

    def interpolate_brightness(self, Rpts, I=None):
        r"""
        Interpolate the brightness profile to the requested set of points.

        The interpolation is done using the Fourier-Bessel series. 
        
        Parameters
        ----------
        Rpts : array, unit = arcsec
            The points to interpolate the brightness to.
        I : array, optional, unit = Jy / Sr
            Intensity points to interpolate. If not specified, the MAP/mean
            will be used. The intensity should be specified at the collocation
            points, I[k] = I(r_k).

        Returns
        -------
        I(Rpts) : array, unit = Jy / Sr
            Brightness at the radial points.

        Notes
        -----
        The resulting brightness will be consistent with higher-resolution fits
        as long as the original fit has sufficient resolution. By sufficient
        resolution we simply mean that the missing terms in the Fourier-Bessel
        series are negligible, which will typically be the case if the
        brightness was obtained from a frank fit with 100 points or more.
        """
        Rpts = np.array(Rpts)
        if I is None:
            I = self.I
        
        V = self._vis_map.transform(I, direction='forward')
        I = self._vis_map.transform(V, Rpts, direction='backward')
        
        if np.any(Rpts > self.Rmax):
            I[Rpts > self.Rmax] = 0

        return I 

    @abc.abstractproperty
    def MAP(self):
        pass

    @property
    def I(self):
        return self.MAP

    @property
    def r(self):
        """Radius points, unit = arcsec"""
        return self._vis_map.r

    @property
    def Rmax(self):
        """Maximum radius, unit = arcsec"""
        return self._vis_map.Rmax
    @property
    def q(self):
        r"""Frequency points, unit = :math:`\lambda`"""
        return self._vis_map.q

    @property
    def Qmax(self):
        r"""Maximum frequency, unit = :math:`\lambda`"""
        return self._vis_map.Qmax

    @property
    def size(self):
        """Number of points in reconstruction"""
        return self._vis_map.size

    @property
    def geometry(self):
        """SourceGeometry object"""
        return self._geometry

    @property
    def info(self):
        """Fit quantities for reference"""
        return self._info


class FrankGaussianFit(FrankRadialFit):
    """
    Result of a frank fit with a Gaussian brightness model.

    Parameters
    ----------
    DHT : DiscreteHankelTransform
        A DHT object with N bins that defines H(p). The DHT is used to compute
        :math:`S(p)`
    fit : GaussianModel object
        Result of fitting with MAP power spectrum.
    info: dict, optional
        Dictionary containing useful quantities for reproducing a fit
        (such as the hyperparameters used)
    geometry: SourceGeometry object, optional
        Geometry used to correct the visibilities for the source
        inclination. If not provided, the geometry determined during the
        fit will be used.
    """

    def __init__(self, DHT, fit, info={}, geometry=None):
        FrankRadialFit.__init__(self, DHT, info, geometry)
        self._fit = fit

    def draw(self, N):
        """Compute N draws from the posterior"""
        return np.random.multivariate_normal(self.mean, self.covariance, N)

    def log_likelihood(self, I=None):
        r"""
        Compute one of two types of likelihood.

        If :math:`I` is provided, this computes

        .. math:
            \log[P(I,V|S)].

        Otherwise the marginalized likelihood is computed,

        .. math:
            \log[P(V|S)].


        Parameters
        ----------
        I : array, size = N, optional, unit = Jy / sr
            Intensity :math:`I(r)` to compute the likelihood of

        Returns
        -------
        log_P : float
            Log likelihood, :math:`\log[P(I,V|p)]` or :math:`\log[P(V|p)]`

        Notes
        -----
        1. The prior probability P(S) is not included.
        2. The likelihoods take the form:

        .. math::
              \log[P(I,V|p)] = \frac{1}{2} j^T I - \frac{1}{2} I^T D^{-1} I
                 - \frac{1}{2} \log[\det(2 \pi S)] + H_0

        and

        .. math::
              \log[P(V|p)] = \frac{1}{2} j^T D j
                 + \frac{1}{2} \log[\det(D)/\det(S)] + H_0

        where

        .. math::
            H_0 = -\frac{1}{2} V^T w V + \frac{1}{2} \sum \log(w /2 \pi)

        is the noise likelihood.
        """
        return self._fit.log_likelihood(I)


    def solve_non_negative(self):
        """Compute the best fit solution with non-negative intensities"""
        return self._fit.solve_non_negative()

    @property
    def mean(self):
        """Posterior mean, unit = Jy / sr"""
        return self._fit.mean

    @property
    def MAP(self):
        """Posterior maximum, unit = Jy / sr"""
        return self.mean

    @property
    def covariance(self):
        """Posterior covariance, unit = (Jy / sr)**2"""
        return self._fit.covariance

    @property
    def power_spectrum(self):
        """Power spectrum coefficients"""
        return self._fit.power_spectrum



class FrankLogNormalFit(FrankRadialFit):
    """
    Result of a frank fit with a Gaussian brightness model.

    Parameters
    ----------
    DHT : DiscreteHankelTransform
        A DHT object with N bins that defines H(p). The DHT is used to compute
        :math:`S(p)`
    fit : LogNormalMAPModel object
        Result of fitting with MAP power spectrum.
    info: dict, optional
        Dictionary containing useful quantities for reproducing a fit
        (such as the hyperparameters used)
    geometry: SourceGeometry object, optional
        Geometry used to correct the visibilities for the source
        inclination. If not provided, the geometry determined during the
        fit will be used.
    """

    def __init__(self, DHT, fit, info={}, geometry=None):
        FrankRadialFit.__init__(self, DHT, info, geometry)
        self._fit = fit

    def log_likelihood(self, I=None):
        r"""
        Compute the likelihood,

        .. math:
            \log[P(I,V|S)].

        Parameters
        ----------
        I : array, size = N, optional
            Intensity :math:`I(r)=exp(s0*s)` to compute the likelihood of

        Returns
        -------
        log_P : float
            Log likelihood, :math:`\log[P(I,V|p)]`

        Notes
        -----
        1. The prior probability P(S) is not included.
        2. The likelihood takes the form:

        .. math::
              \log[P(I,V|p)] = j^T I - \frac{1}{2} I^T D^{-1} I
                 - \frac{1}{2} \log[\det(2 \pi S)] + H_0

        where

        .. math::
            H_0 = -\frac{1}{2} V^T w V + \frac{1}{2} \sum \log(w /2 \pi)

        is the noise likelihood.
        """
        
        if I is not None:
            return self._fit.log_likelihood(np.log(I))
        else:
            return self._fit.log_likelihood()

    @property
    def MAP(self):
        """Posterior maximum, unit = Jy / sr"""
        return np.exp((self._fit.MAP + self._fit.s_0) * self._fit.scale)

    @property
    def covariance(self):
        """Posterior covariance, unit = log[(Jy / sr)**2]"""
        return self._fit.covariance

    @property
    def power_spectrum(self):
        """Power spectrum coefficients"""
        return self._fit.power_spectrum


class FourierBesselFitter(object):
    """
    Fourier-Bessel series model for fitting visibilities

    Parameters
    ----------
    Rmax : float, unit = arcsec
        Radius of support for the functions to transform, i.e.,
            f(r) = 0 for R >= Rmax
    N : int
        Number of collocation points
    geometry : SourceGeometry object
        Geometry used to deproject the visibilities before fitting
    nu : int, default = 0
        Order of the discrete Hankel transform (DHT)
    block_data : bool, default = True
        Large temporary matrices are needed to set up the data. If block_data
        is True, we avoid this, limiting the memory requirement to block_size
        elements.
    assume_optically_thick : bool, default = True
        Whether to correct the visibility amplitudes by a factor of
        1 / cos(inclination); see frank.geometry.rescale_total_flux
    scale_height : function R --> H, optional
        Specifies the vertical thickness of disc as a function of radius. Both
        R and H should be in arcsec. Assumes a Gaussian vertical structure. 
        Only works with assume_optically_thick=False
    block_size : int, default = 10**5
        Size of the matrices if blocking is used
    verbose : bool, default = False
        Whether to print notification messages
    """

    def __init__(self, Rmax, N, geometry, nu=0, block_data=True,
                 assume_optically_thick=True, scale_height=None,
                 block_size=10 ** 5, verbose=True):

        Rmax /= rad_to_arcsec

        self._geometry = geometry

        self._DHT = DiscreteHankelTransform(Rmax, N, nu)

        if assume_optically_thick:
            if scale_height is not None:
                raise ValueError("Optically thick models must have zero "
                                 "scale-height")
            model = 'opt_thick'
        elif scale_height is not None:
            model = 'debris'
        else:
            model = 'opt_thin'

        self._vis_map = VisibilityMapping(self._DHT, geometry, 
                                          model, scale_height=scale_height,
                                          block_data=block_data, block_size=block_size,
                                          check_qbounds=False, verbose=verbose)

        self._info  = {'Rmax' : self._DHT.Rmax * rad_to_arcsec,
                       'N' : self._DHT.size
                       }

        self._verbose = verbose

    def preprocess_visibilities(self, u, v, V, weights=1):
        r"""Prepare the visibilities for fitting. 
        
        This step will be done by the automatically fit method, but it can be 
        expensive. This method is provided to enable the pre-processing to be
        done once when conducting multiple fits with different hyperprior 
        parameters. 

        Parameters
        ----------
        u,v : 1D array, unit = :math:`\lambda`
            uv-points of the visibilies
        V : 1D array, unit = Jy
            Visibility amplitudes at q
        weights : 1D array, optional, unit = Jy^-2
            Weights of the visibilities, weight = 1 / sigma^2, where sigma is
            the standard deviation
        
        Returns
        -------
        processed_visibilities : dict
            Processed visibilities needed in subsequent steps of the fit

        Notes
        -----
        Re-using the processed visibilities is only valid with certain parameter
        changes. For example N, Rmax, geometry, nu, assume_optically_thick, and
        scale_height cannot be changed. This will be checked before fits are 
        conducted.
        """
        return self._vis_map.map_visibilities(u, v, V, weights)

    def _build_matrices(self, mapping):
        r"""
        Compute the matrices M and j from the visibility data.

        Also compute
        .. math:
            `H0 = 0.5*\log[det(weights/(2*np.pi))]
             - 0.5*np.sum(V * weights * V):math:`
        """
        self._vis_map.check_hash(mapping['hash'])
  
        self._M = mapping['M']
        self._j = mapping['j']

        self._H0 = mapping['null_likelihood']

    def fit_method(self):
        """Name of the fit method"""
        return type(self).__name__

    def fit_preprocessed(self, preproc_vis):
        r"""
        Fit the pre-processed visibilties. The last step in the fitting 
        procedure.

        Parameters
        ----------
        preproc_vis : pre-processed visibilities
            Visibilities to fit that have been processed by
            `self.preprocess_visibilities`.

        Returns
        -------
        sol : FrankRadialFit
            Least-squares Fourier-Bessel series fit
        """
        if self._verbose:
            logging.info('  Fitting pre-processed visibilities for brightness'
                         ' profile using {}'.format(self.fit_method()))

        self._build_matrices(preproc_vis)

        return self._fit()

    def fit(self, u, v, V, weights=1):
        r"""
        Fit the visibilties

        Parameters
        ----------
        u,v : 1D array, unit = :math:`\lambda`
            uv-points of the visibilies
        V : 1D array, unit = Jy
            Visibility amplitudes at q
        weights : 1D array, optional, unit = J^-2
            Weights of the visibilities, weight = 1 / sigma^2, where sigma is
            the standard deviation

        Returns
        -------
        sol : FrankRadialFit
            Least-squares Fourier-Bessel series fit
        """
        if self._verbose:
            logging.info('  Fitting for brightness profile using'
                         ' {}'.format(self.fit_method()))

        self._geometry.fit(u, v, V, weights)

        mapping = self.preprocess_visibilities(u, v, V, weights)
        self._build_matrices(mapping)

        return self._fit()

    def _fit(self):
        """Fit step. Computes the best fit given the pre-processed data"""
        fit = GaussianModel(self._DHT, self._M, self._j,
                            noise_likelihood=self._H0)

        self._sol = FrankGaussianFit(self._vis_map, fit, self._info,
                                     geometry=self._geometry.clone())

        return self._sol


    @property
    def r(self):
        """Radius points, unit = arcsec"""
        return self._DHT.r * rad_to_arcsec

    @property
    def Rmax(self):
        """Maximum radius, unit = arcsec"""
        return self._DHT.Rmax * rad_to_arcsec

    @property
    def q(self):
        r"""Frequency points, unit = :math:`\lambda`"""
        return self._DHT.q

    @property
    def Qmax(self):
        r"""Maximum frequency, unit = :math:`\lambda`"""
        return self._DHT.Qmax

    @property
    def size(self):
        """Number of points in reconstruction"""
        return self._DHT.size

    @property
    def geometry(self):
        """Geometry object"""
        return self._geometry


class FrankFitter(FourierBesselFitter):
    """
    Fit a Gaussian process model using the Discrete Hankel Transform of
    Baddour & Chouinard (2015).

    The GP model is based upon Oppermann et al. (2013), which use a maximum
    a posteriori estimate for the power spectrum as the GP prior for the
    real-space coefficients

    Parameters
    ----------
    Rmax : float, unit = arcsec
        Radius of support for the functions to transform, i.e., f(r) = 0 for
        R >= Rmax.
    N : int
        Number of collaction points
    geometry : SourceGeometry object
        Geometry used to deproject the visibilities before fitting
    nu : int, default = 0
        Order of the discrete Hankel transform, given by J_nu(r)
    block_data : bool, default = True
        Large temporary matrices are needed to set up the data. If block_data
        is True, we avoid this, limiting the memory requirement to block_size
        elements
    block_size : int, default = 10**5
        Size of the matrices if blocking is used
    alpha : float >= 1, default = 1.05
        Order parameter of the inverse gamma prior for the power spectrum
        coefficients
    p_0 : float >= 0, default = None, unit=Jy^2
        Scale parameter of the inverse gamma prior for the power spectrum
        coefficients. If not provided p_0 = 1e-15 (method="Normal") or 
        1e-35 (method="LogNormal") will be used.
    weights_smooth : float >= 0, default = 1e-4
        Spectral smoothness prior parameter. Zero is no smoothness prior
    tol : float > 0, default = 1e-3
        Tolerence for convergence of the power spectrum iteration
    method : string, default="Normal"
        Model used for the brightness reconstrution. This must be one of
        "Normal" of "LogNormal".
    I_scale : float, default = 1e5, unit= Jy/Sr
        Brightness scale. Only used in the LogNormal model. Note the
        LogNormal model produces I(Rmax) =  I_scale.
    max_iter: int, default = 2000
        Maximum number of fit iterations
    check_qbounds: bool, default = True
        Whether to check if the first (last) collocation point is smaller
        (larger) than the shortest (longest) deprojected baseline in the dataset
    store_iteration_diagnostics: bool, default = False
        Whether to store the power spectrum parameters and brightness profile
        for each fit iteration
    assume_optically_thick : bool, default = True
        Whether to correct the visibility amplitudes by a factor of
        1 / cos(inclination); see frank.geometry.rescale_total_flux
    scale_height : function R --> H, optional
        Specifies the vertical thickness of disc as a function of radius. Both
        R and H should be in arcsec. Assumes a Gaussian vertical structure. 
        Only works with assume_optically_thick=False
    verbose:
        Whether to print notification messages

    References
    ----------
        Baddour & Chouinard (2015)
            DOI: https://doi.org/10.1364/JOSAA.32.000611
        Oppermann et al. (2013)
            DOI:  https://doi.org/10.1103/PhysRevE.87.032136
    """

    def __init__(self, Rmax, N, geometry, nu=0, block_data=True,
                 block_size=10 ** 5, alpha=1.05, p_0=None, weights_smooth=1e-4,
                 tol=1e-3, method='Normal', I_scale=1e5, max_iter=2000, check_qbounds=True,
                 store_iteration_diagnostics=False, assume_optically_thick=True,
                 scale_height=None, verbose=True):

        if method not in {'Normal', 'LogNormal'}:
            raise ValueError('FrankFitter supports following mehods:\n\t'
                             '{ "Normal", "LogNormal"}"')
        self._method = method

        super(FrankFitter, self).__init__(Rmax, N, geometry, nu, block_data,
                                          assume_optically_thick, scale_height,
                                          block_size, verbose)

        # Reinstate the bounds check: FourierBesselFitter does not check bounds
        self._vis_map.check_qbounds=check_qbounds

        if p_0 is None:
            if method == 'Normal':
                p_0 = 1e-15
            else:
                p_0 = 1e-35

        self._s_scale = np.log(I_scale)

        self._filter = CriticalFilter(
            self._DHT, alpha, p_0, weights_smooth, tol
        )

        self._max_iter = max_iter

        self._store_iteration_diagnostics = store_iteration_diagnostics

        self._info.update({'alpha' : alpha, 'wsmooth' : weights_smooth, 'p0' : p_0})

    def fit_method(self):
        """Name of the fit method"""
        return '{}: {} method'.format(type(self).__name__, self._method)

    def _fit(self):
        """Fit step. Computes the best fit given the pre-processed data"""
        
        if self._store_iteration_diagnostics:
            self._iteration_diagnostics = defaultdict(list)
        
        # Inital guess for power spectrum
        pI = np.ones([self.size])

        # Do an extra iteration based on a power-law guess
        fit = self._perform_fit(pI, guess=np.ones_like(pI), fit_method='Normal')

        pI = np.max(self._DHT.transform(fit.MAP)**2)
        pI *= (self.q / self.q[0])**-2

        fit = self._perform_fit(pI, fit_method='Normal')

        # Now that we've got a reasonable initial brightness, setup the
        # log-normal power spectrum estimate
        if self._method == 'LogNormal':
            s = np.log(np.maximum(fit.MAP, 1e-3 * fit.MAP.max())) 
            s -= self._s_scale
            
            pI = np.max(self._DHT.transform(s)**2)
            pI *= (self.q / self.q[0])**-4

            fit = self._perform_fit(pI, guess=s)

        

        count = 0
        pi_old = 0
        while (not self._filter.check_convergence(pI, pi_old) and
               count <= self._max_iter):

            if self._verbose and logging.getLogger().isEnabledFor(logging.INFO):
                print('\r    {} iteration {}'.format(type(self).__name__, count),
                      end='', flush=True)

            pi_old = pI.copy()
            pI = self._filter.update_power_spectrum(fit)

            fit = self._perform_fit(pI, guess=fit.MAP)

            if self._store_iteration_diagnostics:
                self._iteration_diagnostics['power_spectrum'].append(pI)
                self._iteration_diagnostics['MAP'].append(fit.MAP)

            count += 1

        if self._verbose and logging.getLogger().isEnabledFor(logging.INFO):
            print()

            if count < self._max_iter:
                logging.info('    Convergence criterion met at iteration'
                             ' {}'.format(count-1))
            else:
                logging.info('    Convergence criterion not met; fit stopped at'
                             ' max_iter specified in your parameter file,'
                             ' {}'.format(self._max_iter))

        if self._store_iteration_diagnostics:
            self._iteration_diagnostics['num_iterations'] = count

        # Save the best fit
        if self._method == "Normal":
            self._sol = FrankGaussianFit(self._vis_map, fit, self._info,
                                         geometry=self._geometry.clone())
        else:
            self._sol = FrankLogNormalFit(self._vis_map, fit, self._info,
                                          geometry=self._geometry.clone())

        # Compute the power spectrum covariance at the maximum
        self._ps = pI
        self._ps_cov = None

        return self._sol

    def draw_powerspectrum(self, Ndraw=1):
        """
        Draw N sets of power-spectrum parameters.

        The draw is taken from the Laplace-approximated (Gaussian) posterior
        distribution for p,
            :math:`P(p) ~ G(p - p_MAP, p_cov)`

        Parameters
        ----------
        Ndraw : int, default = 1
            Number of draws

        Returns
        -------
        p : array, size = (N, Ndraw)
            Power spectrum draws
        """

        log_p = np.random.multivariate_normal(np.log(self._ps),
                                              self._ps_cov, Ndraw)

        return np.exp(log_p)

    def _perform_fit(self, p, guess=None, fit_method=None):
        """
        Find the posterior mean and covariance given p

        Parameters
        ----------
        p : array, size = N
            Power spectrum parameters
        guess : array, size = N, option
            Initial guess for brightness used in the fit.
        fit_method : string, optional.
            One of {"Normal", "LogNormal"}. Brightness profile
            method used in fit.

        Returns
        -------
        sol : FrankRadialFit
            Posterior solution object for P(I|V,p)
        """
        if fit_method is None:
            fit_method = self._method

        if fit_method == 'Normal':
            return GaussianModel(self._DHT, self._M, self._j, p,
                                 guess=guess,
                                 noise_likelihood=self._H0)
        elif fit_method == 'LogNormal':
            return LogNormalMAPModel(self._DHT, self._M, self._j, p,
                                     guess=guess, s0=self._s_scale,
                                     noise_likelihood=self._H0)
        else:
            raise ValueError('fit_method must be one of the following:\n\t'
                             '{"Normal", "LogNormal"}')

    def log_prior(self, p=None):
        """
        Compute the log prior probability, log(P(p)),

        .. math:
            `log[P(p)] ~ np.sum(p0/pi - alpha*np.log(p0/pi))
            - 0.5*np.log(p) (weights_smooth*T) np.log(p)`

        Parameters
        ----------
        p : array, size = N, optional
            Power spectrum coefficients. If not provided, the MAP values are
            used

        Returns
        -------
        log[P(p)] : float
            Log prior probability

        Notes
        -----
        Computed up to a normalizing constant that depends on alpha and p0
        """

        if p is None:
            p = self._ps

        return self._filter.log_prior(p)


    def log_likelihood(self, sol=None):
        r"""
        Compute the log likelihood :math:`log[P(p, V)]`,

        .. math:
           \log[P(p, V)] ~ \log[P(V|p)] + \log[P(p)]


        Parameters
        ----------
        sol : FrankRadialFit object, optional
           Posterior solution given a set power spectrum parameters, :math:`p`.
           If not provided, the MAP solution will be used

        Returns
        -------
        log[P(p, V)] : float
            Log prior probability

        Notes
        -----
        Computed up to a normalizing constant that depends on alpha and p0
        """

        if sol is None:
            sol = self.MAP_solution

        return self.log_prior(sol.power_spectrum) + sol.log_likelihood()

    @property
    def MAP_solution(self):
        """Reconstruction for the maximum a posteriori power spectrum"""
        return self._sol

    @property
    def MAP_spectrum(self):
        """Maximum a posteriori power spectrum"""
        return self._ps

    @property
    def MAP_spectrum_covariance(self):
        """Covariance matrix of the maximum a posteriori power spectrum"""
        if self._ps_cov is None:
            self._ps_cov = self._filter.covariance_MAP(self._sol)

        return self._ps_cov

    @property
    def iteration_diagnostics(self):
        """Dict containing power spectrum parameters and posterior mean
        brightness profile at each fit iteration, and number of iterations"""
        return self._iteration_diagnostics
