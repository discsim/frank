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
from frank.statistical_models import GaussianModel, LogNormalMAPModel


class FrankRadialFit(metaclass=abc.ABCMeta):
    """
    Base class for results of frank fits.

    Parameters
    ----------
    DHT : DiscreteHankelTransform
        A DHT object with N bins that defines H(p). The DHT is used to compute
        :math:`S(p)`
    geometry: SourceGeometry object, optional
        Geometry used to correct the visibilities for the source
        inclination. If not provided, the geometry determined during the
        fit will be used.
    """
    def __init__(self, DHT, geometry):
        self._DHT = DHT
        self._geometry = geometry

    def _predict(self, q, I, block_size):
        """Perform the visibility prediction"""

        if q is None:
            q = self.q

        if I is None:
            I = self.MAP
        
        # Block the visibility calulation for speed
        Ni = int(block_size / len(I) + 1)

        end = 0
        start = 0
        V = []
        while end < len(q):
            start = end
            end = start + Ni
            qi = q[start:end]

            V.append(self._DHT.transform(I, qi))

        return np.concatenate(V)

    def predict(self, u, v, I=None, geometry=None, block_size=10**5):
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
        block_size : int, default = 10**5
            Maximum matrix size used in the visibility calculation

        Returns
        -------
        V(u,v) : array, unit = Jy
            Predicted visibilties of a source with a radial flux distribution
            given by :math:`I` and the position angle, inclination and phase
            centre determined by the geometry object
        """
        if geometry is None:
            geometry = self._geometry

        if geometry is not None:
            u, v = self._geometry.deproject(u, v)

        q = np.hypot(u, v)
        V = self._predict(q, I, block_size)

        if geometry is not None:
            V *= np.cos(geometry.inc * deg_to_rad)

        # Undo phase centering
        _, _, V = geometry.undo_correction(u, v, V)

        return V

    def predict_deprojected(self, q=None, I=None, geometry=None,
                            block_size=10**5):
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
        for consistentcy with `uvplot`
        """
        if geometry is None:
            geometry = self._geometry

        V = self._predict(q, I, block_size)

        if geometry is not None:
            V *= np.cos(geometry.inc * deg_to_rad)

        return V

    @abc.abstractproperty
    def MAP(self):
        pass

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
        return self._geometry


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
    geometry: SourceGeometry object, optional
        Geometry used to correct the visibilities for the source
        inclination. If not provided, the geometry determined during the
        fit will be used.
    """

    def __init__(self, DHT, fit, geometry=None):
        FrankRadialFit.__init__(self, DHT, geometry)
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
    geometry: SourceGeometry object, optional
        Geometry used to correct the visibilities for the source
        inclination. If not provided, the geometry determined during the
        fit will be used.
    """

    def __init__(self, DHT, fit, geometry=None):
        FrankRadialFit.__init__(self, DHT, geometry)
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
        return np.exp(self._fit.MAP * self._fit.scale)

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
    block_size : int, default = 10**5
        Size of the matrices if blocking is used
    verbose : bool, default = False
        Whether to print notification messages
    """

    def __init__(self, Rmax, N, geometry, nu=0, block_data=True,
                 block_size=10 ** 5, verbose=True):

        Rmax /= rad_to_arcsec

        self._geometry = geometry

        self._DHT = DiscreteHankelTransform(Rmax, N, nu)

        self._blocking = block_data
        self._block_size = block_size

        self._verbose = verbose

    def _check_uv_range(self, uv):
        """Don't check the bounds for FourierBesselFitterr"""
        pass

    def _build_matrices(self, u, v, V, weights):
        r"""
        Compute the matrices M and j from the visibility data.

        Also compute
        .. math:
            `H0 = 0.5*\log[det(weights/(2*np.pi))]
             - 0.5*np.sum(V * weights * V):math:`
        """
        if self._verbose:
            logging.info('    Building visibility matrices M and j')

        # Deproject the visibilities
        u, v, V = self._geometry.apply_correction(u, v, V)
        q = np.hypot(u, v)

        # Check consistency of the uv points with the model
        self._check_uv_range(q)

        # Use only the real part of V. Also correct the total flux for the
        # inclination. This is not done in apply_correction for consistency
        # with `uvplot`
        V = V.real / np.cos(self._geometry.inc * deg_to_rad)
        weights = weights * np.cos(self._geometry.inc * deg_to_rad) ** 2

        # If blocking is used, we will build up M and j chunk-by-chunk
        if self._blocking:
            Nstep = int(self._block_size / self.size + 1)
        else:
            Nstep = len(V)

        # Ensure the weights are 1D
        w = np.ones_like(V) * weights

        start = 0
        end = Nstep
        Ndata = len(V)
        M = 0
        j = 0
        while start < Ndata:
            qs = q[start:end]
            ws = w[start:end]
            Vs = V[start:end]

            X = self._DHT.coefficients(qs)

            wXT = np.array(X.T * ws, order='C')

            M += np.dot(wXT, X)
            j += np.dot(wXT, Vs)

            start = end
            end += Nstep

        self._M = M
        self._j = j

        # Compute likelihood normalization H_0
        self._H0 = 0.5 * np.sum(np.log(w / (2 * np.pi)) - V * w * V)

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
                         ' FourierBesselFitter')

        self._geometry.fit(u, v, V, weights)

        self._build_matrices(u, v, V, weights)

        fit = GaussianModel(self._DHT, self._M, self._j,noise_likelihood=self._H0)

        self._sol = FrankGaussianFit(self._DHT, fit,
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
    max_iter: int, default = 2000
        Maximum number of fit iterations
    check_qbounds: bool, default = True
        Whether to check if the first (last) collocation point is smaller
        (larger) than the shortest (longest) deprojected baseline in the dataset
    store_iteration_diagnostics: bool, default = False
        Whether to store the power spectrum parameters and brightness profile
        for each fit iteration
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
                 tol=1e-3, method='Normal', max_iter=2000, check_qbounds=True,
                 store_iteration_diagnostics=False, verbose=True):

        if method not in {'Normal', 'LogNormal'}:
            raise ValueError('FrankFitter supports following mehods:\n\t'
                             '{ "Normal", "LogNormal"}"')
        self._method = method

        super(FrankFitter, self).__init__(Rmax, N, geometry, nu, block_data,
                                          block_size, verbose)

        if p_0 is None:
            if method == 'Normal':
                p_0 = 1e-15
            else:
                p_0 = 1e-35

        self._filter = CriticalFilter(
            self._DHT, alpha, p_0, weights_smooth, tol
        )

        self._max_iter = max_iter

        self._check_qbounds = check_qbounds
        self._store_iteration_diagnostics = store_iteration_diagnostics

    def _check_uv_range(self, uv):
        """Check that the uv domain is properly covered"""

        # Check whether the first (last) collocation point is smaller (larger)
        # than the shortest (longest) deprojected baseline in the dataset
        if self._check_qbounds:
            if self.q[0] < uv.min():
                logging.warning(r"WARNING: First collocation point, q[0] = {:.3e} \lambda,"
                                " is at a baseline shorter than the"
                                " shortest deprojected baseline in the dataset,"
                                r" min(uv) = {:.3e} \lambda. For q[0] << min(uv),"
                                " the fit's total flux may be biased"
                                " low.".format(self.q[0], uv.min()))

            if self.q[-1] < uv.max():
                raise ValueError(r"ERROR: Last collocation point, {:.3e} \lambda, is at"
                                 " a shorter baseline than the longest deprojected"
                                 r" baseline in the dataset, {:.3e} \lambda. Please"
                                 " increase N in FrankFitter (this is"
                                 " `hyperparameters: n` if you're using a parameter"
                                 " file). Or if you'd like to fit to shorter maximum baseline,"
                                 " cut the (u, v) distribution before fitting"
                                 " (`modify_data: baseline_range` in the"
                                 " parameter file).".format(self.q[-1], uv.max()))


    def fit(self, u, v, V, weights=1):
        r"""
        Fit the visibilties

        Parameters
        ----------
        u,v : 1D array, unit = :math:`\lambda`
            uv-points of the visibilies
        V : 1D array, unit = Jy
            Visibility amplitudes at q
        weights : 1D array, optional, unit = Jy^-2
            Weights of the visibilities, weight = 1 / sigma^2, where sigma is
            the standard deviation
        iteration_diagnostics : dict, optional,
          size = N_iter x 2 x N_{collocation points}
          Power spectrum parameters and posterior mean brightness profile at
          each fit iteration, and number of iterations

        Returns
        -------
        MAP_solution : FrankRadialFit
            Reconstructed profile using maximum a posteriori power spectrum
        """
        if self._verbose:
            logging.info('  Fitting for brightness profile using FrankFitter')

        if self._store_iteration_diagnostics:
            self._iteration_diagnostics = defaultdict(list)

        # Fit geometry if needed
        self._geometry.fit(u, v, V, weights)


        # Project the data to the signal space
        self._build_matrices(u, v, V, weights)
     
        # Inital guess for power spectrum
        pi = np.ones([self.size])

        # Do an extra iteration based on a power-law guess
        fit = self._perform_fit(pi, fit_method='Normal')
   
        pi = np.max(self._DHT.transform( fit.MAP)**2)
        pi *= (self.q / self.q[0])**-2

        fit = self._perform_fit(pi, fit_method='Normal')

        # Now that we've got a reasonable initial brightness, setup the
        # log-normal power spectrum estimate
        if self._method == 'LogNormal':
            s = np.log(np.maximum(fit.MAP, 1e-3 * fit.MAP.max()))
            
            pi = np.max(self._DHT.transform(s)**2)
            pi *= (self.q / self.q[0])**-4

            fit = self._perform_fit(pi, guess=s)

        

        count = 0
        pi_old = 0
        while (not self._filter.check_convergence(pi, pi_old) and
               count <= self._max_iter):

            if self._verbose and logging.getLogger().isEnabledFor(logging.INFO):
                print('\r    FrankFitter iteration {}'.format(count),
                      end='', flush=True)

            pi_old = pi.copy()
            pi = self._filter.update_power_spectrum(fit)

            fit = self._perform_fit(pi, guess=fit.MAP)

            if self._store_iteration_diagnostics:
                self._iteration_diagnostics['power_spectrum'].append(pi)
                self._iteration_diagnostics['mean'].append(fit.MAP)

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
            self._sol = FrankGaussianFit(self._DHT, fit,
                                         geometry=self._geometry.clone())
        else:
            self._sol = FrankLogNormalFit(self._DHT, fit,
                                          geometry=self._geometry.clone())

        # Compute the power spectrum covariance at the maximum
        self._ps = pi
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
                                     guess=guess,
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
           If not provided, the MAP solution will be provided

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
