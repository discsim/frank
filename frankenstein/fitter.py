import numpy as np
from scipy.special import j0, j1, jn_zeros, jv
import scipy.linalg
#from scipy.integrate import quad, simps
#from scipy.interpolate import PchipInterpolator, interp1d
from scipy.stats import norm#, multivariate_normal, gamma, invgamma
#from scipy.optimize import minimize
#from scipy.linalg import blas # TODO: clean up
import time

class GPHankelFitter(object):
    '''Fit a Gaussian process (GP) model using the Discrete Hankel Transform
    (DHT) of Baddour & Chouinard (2015; referenced here as BC15).

    The GP model is based upon Oppermann et al. (2013; referenced as OSBE13),
    which use a maximum a posteriori estimate for the power spectrum as the
    GP prior for the real space coefficients.

    args:
        Rmax : float, default precomputed # TODO
            Radius of support for the functions to transform, i.e.,
              f(r) = 0 for R >= Rmax
            By default this is computed by xx # TODO
        N : int
            Number of collocation points
        alpha : float >= 1, default = 1.05 # TODO: update default (this and others)
            Order parameter of the inverse gamma prior for the power spectrum
            coefficients
        q : float >= 0, default = 1e-12
            Scale parameter of the inverse gamma prior for the power spectrum
            coefficients
        tol : float > 0, default = 1e-10
            Tolerence for convergence of the power spectrum iteration
        enforce_postive : bool, default = False
            Determines whether the 'mean' function evaluated is the true
            mean of the posterior (for the assumed power spectrum), or
            the best fit subject to the constraint that the reconstructed
            function is positive semi-definite.
    ======

    Note:
        If the transform is only needed at the collocation points, the methods
        forward_transform and inverse_transform are more expedient than using
        the HankelTransform method.

    Reference:
        Baddour & Chouinard (2015)
          DOI: https://doi.org/10.1364/JOSAA.32.000611
        Oppermann et al. (2013)
          DOI: https://doi.org/10.1103/PhysRevE.87.032136
    '''

    def __init__(self, Rmax, N=128, alpha=1.1, q=1e-12, tol=1e-10,
                 enforce_positive=False):
        ## Compute the collocation points
        # TODO: set N as precomputed by default
        j_nk = jn_zeros(0, N + 1) # The k-th root of the 0th order Bessel. For
        #visibilities, by definition 0 is the order of the DHT
        j_nk, j_nN = j_nk[:-1], j_nk[-1]
        Wmax = j_nk[-1] / Rmax
        # TODO: why no 2\pi out front? should j_nk[-1] be j_nN?
        self._Rnk = Rmax * (j_nk / j_nN) # Collocation points in real space
        self._Wnk = Wmax * (j_nk / j_nN) # Corresponding points in spatial
        # frequency space
        t0=time.time()
        print('t0',t0)
        ## Compute the kernel matrix used in the transform
        Jnk = np.outer(np.ones_like(j_nk), j1(j_nk)) # Amplitudes of
        # J_1 at the roots of J_0. In BC15 this is J_{n+1}(j_{nk})
        t1=time.time()
        print('Delta t1',t1-t0)
        Jmk = Jnk # TODO: why do this?
        t2=time.time()
        print('Delta t2',t2-t1)

        self._Ymk = (2 / (j_nN * Jmk * Jnk)) * \
            j0(np.prod(np.meshgrid(j_nk, j_nk / j_nN), axis=0)) # Kernel
        # for the DHT (BC15 eqn.19) # TODO: I renamed Ykm --> Ymk (for the forward transform)
        t3=time.time()
        print('Delta t3',t3-t2)
        self._scale_factor = 2 / j1(j_nk) ** 2 # Amplitude scaling of
        # the Bessel functions used in the fit (see BC15 eqn.11)
        t3_2=time.time()
        print('Delta t3_2',t3_2-t3)

        ## Store the remaining needed parameters
        self._j_nk = j_nk
        self._j_nN = j_nN

        self._Rmax = Rmax
        self._Wmax = Wmax

        self._qi = q
        self._ai = alpha

        self._tol = tol

        self._enforce_positive = enforce_positive

    def _hankel_coeffs(self, k):
        '''Discrete Hankel coefficients for a forward transform'''
        tj0 = time.time()
        #j0(np.outer(k / self._Wmax, self._j_nk))
        tj1 = time.time()
        print('Delta tj1',tj1-tj0)
        H = (self._scale_factor / self._Wmax ** 2) * j0(np.outer(k / self._Wmax, self._j_nk)) ## BC15 eqn.11?
        ## BC15 eqn.5 (generalized eqn.11) to evaluate DHT at given freqs

        return 2 * np.pi * H # TODO: why is 2 * np.pi included here and in other 'return's? did we miss it somewhere more internal?

    def _fit_data(self, uv, yi, w=1.): # uv are baselines, yi are uv.re, w are weights
        '''Fit the Hankel coefficients of the transform'''
        print('len uv',len(uv))
        k = 2 * np.pi * uv

        ta=time.time()
        print('ta',ta)
        H = self._hankel_coeffs(k)
        tb=time.time()
        print('Delta tb',tb-ta)

        wHT = np.array(H.T * w, order='C')
        tb2=time.time()
        print('Delta tb2',tb2-tb)

        self._M = np.dot(wHT, H) # TODO
        tb3=time.time()
        print('Delta tb3',tb3-tb2)
        self._j = np.dot(wHT, yi) # TODO
        tb4=time.time()
        print('Delta tb4',tb4-tb3)

    def _fit_GP(self, covarience, inverse=False, enforce_positive=None):
        '''Fit the power spectrum of the data as a prior for the coefficients # TODO: add sentences describing this. construct D and solve for means...
        Rename vars to be consistent w/ our paper
        use cholesky factorization else use SVD
        ref corresponding eqn #s in our paper''' # TODO: right?
        if not inverse:
            tA=time.time()
            Dinv = self._M + np.linalg.inv(covarience) # least squares w/ extra covar term
            tB=time.time()
            print('Delta tB',tB-tA)
        else:
            Dinv = self._M + covarience

        if enforce_positive is None: enforce_positive = self._enforce_positive
        try:
            tC=time.time()
            self._Dchol = scipy.linalg.cho_factor(Dinv)
            self._Dsvd = None
            tD=time.time()
            #print('Delta tD',tD-tC)

            if not enforce_positive:
                self._mu = scipy.linalg.cho_solve(self._Dchol, self._j)
                tE=time.time()
                #print('Delta tE',tE-tD)
            else:
                self._mu = scipy.optimize.nnls(Dinv, self._j, maxiter=32 * Dinv.shape[1])[0]

        except np.linalg.LinAlgError as e:
            U, s, V = scipy.linalg.svd(Dinv, full_matrices=False)

            print('fit encountering difficulty: ',e) # TODO: write to a logfile
            s1 = np.where(s > 0, 1. / s, 0)

            self._Dchol = None
            self._Dsvd = U, s1, V

            if not enforce_positive:
                self._mu = np.dot(V.T, np.multiply(np.dot(U.T, self._j), s1))
            else:
                self._mu = scipy.optimize.nnls(Dinv, self._j, maxiter=32 * Dinv.shape[1])[0]

        return self._mu

    def _Dsolve(self, x):
        '''# TODO: define'''
        if self._Dchol is not None:
            return scipy.linalg.cho_solve(self._Dchol, x)
        else:
            U, s1, V = self._Dsvd 
            return np.dot(V.T, np.multiply(np.dot(U.T, x), s1))

    def _build_smoothing_matrix(self):
        '''# TODO: define'''
        log_k = np.log(self.kc)
        dc = (log_k[2:] - log_k[:-2]) / 2 # OSRE13 eqn.B2
        de = np.diff(log_k) # OSRE13 eqn.B3

        Delta = np.zeros([3, self.size])
        Delta[0, :-2] = 1 / (dc * de[:-1])
        Delta[1, 1:-1] = - (1 / de[1:] + 1 / de[:-1]) / dc # OSRE13 eqn.B5
        Delta[2, 2:] = 1 / (dc * de[1:]) # OSRE13 eqn.B6

        Delta = scipy.sparse.dia_matrix((Delta, [-1, 0, 1]), shape=(self.size, self.size))

        dce = np.zeros_like(log_k)
        dce[1:-1] = dc
        dce = scipy.sparse.dia_matrix((dce.reshape(1, -1), 0), shape=(self.size, self.size))

        Tij = Delta.T.dot(dce.dot(Delta)) # OSRE13 eqn.B8 sans the 1 / \sigma_p^2
        # prefactor

        return Tij

    def fit(self, k, yi, smooth_strength=1e-4, w=1): ## k are baselines, yi are uv.re, w are weights
        '''# TODO: define'''
        ## Project the data to the signal space
        self._fit_data(k, yi, w)

        ## Compute the factors needed for the kernel iteration
        norm = (self._j_nN / (self._Wmax * self._Wmax)) ** 2
        rho = 1.0

        ## Compute the smoothing matrix:
        Tij = self._build_smoothing_matrix()

        def inv_cov_matrix(p): # pass in pwr spectrum and return cov matrix
            '''Construct inverse covarience matrix'''
            p1 = np.where(p > 0 * p.max(), 1. / p, 0)
            return np.einsum('ji,j,jk->ik', self._Ymk, p1, self._Ymk) * norm

        def cov_matrix(p):
            '''Construct covarience matrix'''
            return np.einsum('ij,j,kj->ik', self._Ymk, p, self._Ymk) / norm

        ## Setup kernel parameters
        pi = np.ones([self.size])
        mu = self._fit_GP(inv_cov_matrix(pi), inverse=True)

        pi[:] = norm * np.max(np.dot(self._Ymk, mu)) ** 2 / (self._ai + 0.5 * rho - 1.0)
        #pi[self.uv_pts > k.max()] *= 1e-10
        #pi[self.uv_pts > k.max()] = 1e-6
        pi *= (self.kc / self.kc[0]) ** -2 # TODO: adjust pi here, i.e., adjust power law for initial power spectrum guess
        #pi *= (self.kc / self.kc[0]) ** -2 # set initial power spectrum guess to constant
        #pi = np.maximum(pi, self._qi)

        mu = self._fit_GP(inv_cov_matrix(pi), inverse=True)

        ## Do one unsmoothed iteration
        Tr1 = norm * np.dot(self._Ymk, mu) ** 2
        #Tr2 = norm * np.einsum('ij,ji->i', self._Ymk, scipy.linalg.cho_solve(self._Dchol, self._Ymk.T))
        Tr2 = norm * np.einsum('ij,ji->i', self._Ymk, self._Dsolve(self._Ymk.T)) # projection of Dkk into pwr spectrum space
        pi = (self._qi + 0.5 * (Tr1 + Tr2)) / (self._ai - 1.0 + 0.5 * rho)

        chi2s = []  # temporary
        chi2s_alt = []  # temporary
        uv_pts_toplot = []
        pi_toplot = []
        alphas_to_plot = []
        mu_to_plot = []
        pointwise_sc_to_plot = []
        stop_criterion_to_plot = []
        condition_numbers = []
        count = 0
        # while count == 0 or (np.any(np.abs(pi-pi_old) > self._tol * np.mean(pi**2)**0.5) and count <= 750):
        while count < 250:
        ##while count < 250:
            print('\r    smoothing fit. iteration %s' % count, end='', flush=True)

            # Project mu to k-space
            #   Tr1 = Trace(mu mu_T . Ymk_T Ymk) = Trace( Ymk mu . (Ymk mu)^T) = (Ymk mu)**2
            Tr1 = norm * np.dot(self._Ymk, mu) ** 2
            # Project D to k-space:
            #   Drr^-1 = Ymk^T Dkk^-1 Ymk
            #   Drr = Ymk^-1 Dkk Ymk^-T
            #   Dkk = Ymk Drr Ymk^T
            # Tr2 = Trace(Dkk)
            #Tr2 = norm * np.einsum('ij,ji->i', self._Ymk, scipy.linalg.cho_solve(self._Dchol, self._Ymk.T))
            Tr2 = norm * np.einsum('ij,ji->i', self._Ymk, self._Dsolve(self._Ymk.T))

            '''
            #smooth =  0 #Tij.dot(np.log(pi))
            smooth = Tij.dot(np.log(pi)) # temporary

            #pi_new =  (2*self._qi + Tr1 + Tr2) / (2*(self._ai-1.0) + rho + smooth)
            #pi_new =  (2*self._qi + Tr1 + Tr2) / (2*(self._ai-1.0) + rho + 1e-8 * count * smooth) # temporary
            pi_new =  (2*self._qi + Tr1 + Tr2) / (2*(self._ai-1.0) + rho) # temporary (no smoothing)
            ##if count == 83:
            ##    pi_new =  (2*self._qi + Tr1 + Tr2) / (2*(self._ai-1.0) + rho + smooth) # temporary
            #print('\ncount',count,'np.mean(smooth)',np.mean(smooth))
            #print('np.mean(pi_new)',np.mean(pi_new))

            tau = scipy.sparse.linalg.spsolve(scipy.sparse.identity(self.size) + smooth_strength *Tij, np.log(pi_new)) # new smoothing approach
            # TODO: adjust 1e-4? e.g., try 1e-1
            pi_new = np.exp(tau)
            '''

            alpha = self._ai + 2 / (1 + .5 * count) # TODO: vary this to adjust how quickly a higher initial alpha
            # (used to damp erroneously high regions of the power spectrum) converges to the input alpha value
            #alpha=1.05
            alphas_to_plot.append(alpha)
            #beta = (self._qi + 0.5 * (Tr1 + Tr2)) / pi - (self._ai - 1.0 + 0.5 * rho)
            beta = (self._qi + 0.5 * (Tr1 + Tr2)) / pi - (alpha - 1.0 + 0.5 * rho)   # TODO: use in place of the above 'smooth = ' ... 'pi_new = '?
            if count == 0: beta = 0
            tau = scipy.sparse.linalg.spsolve(scipy.sparse.identity(self.size) + smooth_strength * Tij, beta + np.log(pi))
            pi_new = np.exp(tau) # power spectral component amplitudes
            #'''

            pi_old = pi.copy()
            pi = pi_new

            # pi[1:-1] = np.maximum(pi_new[:-2], np.maximum(pi_new[1:-1], pi_new[2:]))
            # pi[0]  = max(pi_new[:2])
            # pi[-1] = max(pi_new[-2:])

            mu = self._fit_GP(inv_cov_matrix(pi), inverse=True)
            mu_to_plot.append(mu)

            #icovar = np.linalg.inv(scipy.linalg.cho_solve(self._Dchol, np.eye(self.size)))  # temporary
            '''
            icovar = np.linalg.inv(self._Dsolve(np.eye(self.size)))  # temporary
            chi2_i = np.sum(yi * w * yi) - np.dot(mu, np.dot(icovar, mu))  # temporary # TODO: which dfn of chi2 is right?
            chi2_i_alt = np.sum(yi * w * yi) - np.dot(mu, np.dot(self._M, mu))  # temporary
            chi2s.append(chi2_i)  # temporary
            chi2s_alt.append(chi2_i_alt)  # temporary
            condition_num = np.linalg.cond(icovar) # TODO: use icovar?
            condition_numbers.append(condition_num)
            '''
            #if (count > 0 and np.abs(np.diff(chi2s_alt)[-1]) < 1e-4):
            #    break
            #if count > 5 and chi2s[-1] > chi2s[-2]:
            #    break

            uv_pts_toplot.append(self.uv_pts)
            pi_toplot.append(pi)

            pointwise_sc = ((pi - pi_old) / (pi * .01))**2
            stop_criterion = np.mean(pointwise_sc)
            pointwise_sc_to_plot.append(pointwise_sc)
            stop_criterion_to_plot.append(stop_criterion)
            count += 1

        print('\n    fit complete')
        #self._cov = scipy.linalg.cho_solve(self._Dchol, np.eye(self.size))
        self._cov = self._Dsolve(np.eye(self.size))

        return self._mu, self._cov, uv_pts_toplot, pi_toplot, mu_to_plot, alphas_to_plot, pointwise_sc_to_plot, stop_criterion_to_plot, count  # temporary

    def HankelTransform(self, k):
        '''Compute the hankel tranform of the means at points k'''
        H = self._hankel_coeffs(k)

        return np.dot(H, self.mean)
        # TODO: should this be 'return np.dot(H,self.mu)'?

    def forward_transform(self, f_m):
        '''Compute the forward discrete Hankel transform of fi.

        Uses precomputed coefficients evaluted at points Rc, kc
        '''
        return 2 * np.pi * np.dot(self._Ymk, f_m) * self._j_nN / (self._Wmax * self._Wmax)

    def inverse_transform(self, H_m):
        '''Compute the inverse discrete Hankel transform of fi.

        Uses precomputed coefficients evaluted at points kc, Rc
        '''
        return np.dot(self._Ymk, H_m) * self._j_nN / (self._Rmax * self._Rmax) / (2 * np.pi)

    @property
    def Rc(self):
        '''Real space collocation points used in the fit'''
        return self._Rnk

    @property
    def size(self):
        '''Number of collocation points'''
        return self._Rnk.shape[0]

    @property
    def kc(self):
        '''Spatial frequency space collocation points used in the fit'''
        return self._Wnk

    @property
    def uv_pts(self):
        '''Baselines. units: [cycles / m]'''
        return self.kc / (2 * np.pi)

    @property
    def Rmax(self):
        '''Radius of support for I(r)'''
        return self._Rmax

    @property
    def Wmax(self):
        '''Maximum frequency of support for V(k)'''
        return self._Wmax

    @property
    def Inu(self):
        '''Intensity values at real space collocation points'''
        return self._mu

    @property
    def mean(self):
        '''Mean of the posterior distribution for the intensity profile'''
        return self._mu

    @property
    def cov(self):
        '''Covariance matrix used in the Hankel transform'''
        return self._cov
