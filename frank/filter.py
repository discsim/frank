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
import numpy as np
import scipy.sparse

from frank.constants import rad_to_arcsec


class CriticalFilter:
    def __init__(self, DHT, alpha, p_0, weights_smooth,
                 tol=1e-3):

        self._DHT = DHT

        self._alpha = alpha
        self._p_0 = p_0
        self._rho = 1.0

        self._tol = tol

        self._Tij = weights_smooth * self._build_smoothing_matrix()

    def _build_smoothing_matrix(self):
        log_q = np.log(self._DHT.q)
        dc = (log_q[2:] - log_q[:-2]) / 2
        de = np.diff(log_q)

        N = self._DHT.size

        Delta = np.zeros([3, N])
        Delta[0, :-2] = 1 / (dc * de[:-1])
        Delta[1, 1:-1] = - (1 / de[1:] + 1 / de[:-1]) / dc
        Delta[2, 2:] = 1 / (dc * de[1:])

        Delta = scipy.sparse.dia_matrix((Delta, [-1, 0, 1]),
                                        shape=(N, N))

        dce = np.zeros_like(log_q)
        dce[1:-1] = dc
        dce = scipy.sparse.dia_matrix((dce.reshape(1, -1), 0),
                                      shape=(N, N))

        Tij = Delta.T.dot(dce.dot(Delta))

        return Tij 

    def update_power_spectrum(self, fit):
        """Estimate the best fit power spectrum given the current model fit"""
        Ykm = self._DHT.coefficients()
        Tij_pI = self._Tij + scipy.sparse.identity(self._DHT.size)

        # Project mu to Fourier-space
        #   Tr1 = Trace(mu mu_T . Ykm_T Ykm) = Trace( Ykm mu . (Ykm mu)^T)
        #       = (Ykm mu)**2
        Tr1 = np.dot(Ykm, fit.MAP) ** 2
        # Project D to Fourier-space
        #   Drr^-1 = Ykm^T Dqq^-1 Ykm
        #   Drr = Ykm^-1 Dqq Ykm^-T
        #   Dqq = Ykm Drr Ykm^T
        # Tr2 = Trace(Dqq)
        Tr2 = np.einsum('ij,ji->i', Ykm, fit.Dsolve(Ykm.T))

        pi = fit.power_spectrum

        beta = (self._p_0 + 0.5 * (Tr1 + Tr2)) / pi - \
               (self._alpha - 1.0 + 0.5 * self._rho)
                
        tau = scipy.sparse.linalg.spsolve(Tij_pI, beta + np.log(pi))

        return np.exp(tau)

    def check_convergence(self, pi_new, pi_old):
        """Determine whether the  power-spectrum iterations have converged"""
        return np.all(np.abs(pi_new - pi_old) <= self._tol * pi_new)


    def covariance_MAP(self, fit):
        """
        Covariance of the power spectrum at maximum likelihood

        Parameters
        ----------
        fit : _HankelRegressor
            Solution at maximum likelihood

        Returns
        -------
        ps_cov : 2D array
            Covariance matrix of the power spectrum at maximum likelihood

        Notes
        -----
        Only valid at the location of maximum likelihood

        """
        Ykm = self._DHT.coefficients()

        mq = np.dot(Ykm, fit.MAP)

        mqq = np.outer(mq, mq)
        Dqq = np.dot(Ykm, fit.Dsolve(Ykm.T))

        p = fit.power_spectrum
        tau = np.log(p)

        hess = \
            + np.diag(self._alpha - 1.0 + 0.5 * self._rho + self._Tij.dot(tau)) \
            + self._Tij.todense() \
            - 0.5 * np.outer(1 / p, 1 / p) * (2 * mqq + Dqq) * Dqq

        # Invert the Hessian
        hess_chol = scipy.linalg.cho_factor(hess)
        ps_cov = scipy.linalg.cho_solve(hess_chol, np.eye(self._DHT.size))

        return ps_cov

    def log_prior(self, p):
        """
        Compute the log prior probability, log(P(p)),

        .. math:
            `log[P(p)] ~ np.sum(p0/pi - alpha*np.log(p0/pi))
            - 0.5*np.log(p) (weights_smooth*T) np.log(p)`

        Parameters
        ----------
        p : array, size = N
            Power spectrum coefficients.
        
        Returns
        -------
        log[P(p)] : float
            Log prior probability

        Notes
        -----
        Computed up to a normalizing constant that depends on alpha and p0
        """

        # Add the power spectrum prior term
        xi = self._p_0 / p
        like = np.sum(xi - self._alpha * np.log(xi))

        # Extra term due to spectral smoothness
        tau = np.log(p)
        like -= 0.5 * np.dot(tau, self._Tij.dot(tau))

        return like