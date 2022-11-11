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
"""This module contains methods solving non-linear minimization problems.
"""
import numpy as np
import scipy.linalg

class BaseLineSearch(object):
    """Base class for back-tracking line searches.
    
    Parameters
    ----------
    reduce_step: function (dx, x) --> dx, optional
        Can be provided to limit the maximum step allowed in newton iterations
        given the current position (x) and suggested step (dx). No limiting is
        done if this function is not provided. 
    """

    def __init__(self, reduce_step=None):
        self.reduction = None

        if reduce_step is None:
            self.reduce_step = lambda dx, _: dx
        else:
            self.reduce_step = reduce_step


class LineSearch(BaseLineSearch):
    """Back-tracking line search for Newton's method and gradient descent.

    Iteratively changes the step size used in the update until an acceptable
    reduction in the cost function is found.

    Parameters
    ----------
    armijo : float, default = 1e-4
        Coefficient used when deciding whether to accept a new solution. 
        Smaller is more tolerant.
    min_step_frac : foat, default = 0.1
        Minimum amount that the step length can be reduced by in a single
        iteration when trying to find an improved guess.
    reduce_step: function (dx, x) --> dx, optional
        Can be provided to limit the maximum step allowed in newton iterations
        given the current position (x) and suggested step (dx). No limiting is
        done if this function is not provided. 

    Notes
    -----
    This implementation is baeed on numerical recipes.
    """

    def __init__(self, armijo=1e-4, min_step_frac=0.1, reduce_step=None):
        super(LineSearch, self).__init__(reduce_step)

        self._armijo = armijo
        self._l_min = min_step_frac

    def __call__(self, func, jac, x0, p, f0=None, root=True):
        """Find a good step using backtracking.

        Parameters
        ----------
        func : function,
            The function that we are trying to find the root of
        jac : Jacobian object,
            Must provide a "dot" method that returns the dot-product of the
            Jacobian with a vector.
        x0 : array
            Current best guess for the solution
        p : array
            Step direction.
        f0 : array, optional.
            Evaluation of func at x0, i.e. func(x0). If not provided then this
            will be evaluated internally.
        root : 

        Returns
        -------
        x_new : array
            Recommended new point
        f_new : array
            func(x_new)
        nfev : int
            Number of function evaluations
        failed : bool
            Whether the line-search failed to produce a guess
        """
        def eval(x):
            fvec = func(x)
            if root:
                f = 0.5*np.dot(fvec, fvec)
                return fvec, f
            else:
                return fvec, fvec

        nfev = 0
        if f0 is None:
            f0, cost = eval(x0)
            nfev += 1
        else:
            if root:
                cost = 0.5*np.dot(f0, f0)
            else:
                cost = f0

        # First make sure the step isn't trying to change x by too much
        p = self.reduce_step(p, x0)

        # Compute the expected change in f due to the step p assuming f is
        # exactly described by its linear expansion about the root:
        if root:
            delta_f = np.dot(f0, np.dot(jac, p))
        else:
            delta_f = np.dot(jac, p)

        if delta_f > 0:
            raise ValueError("Round off in slope calculation")

        # Start with the full newton step
        lam = 1.0
        cost_save = lam_save = None
        while True:
   
            x_new = x0 + lam*p
            if np.all(x_new == x0):
                return x0, f0, nfev, True

            f_new, cost_new = eval(x_new)
            nfev += 1

            # Have we got an acceptable step?
            if cost_new <= (cost + self._armijo*lam*delta_f):
                self.reduction = lam
                return x_new, f_new, nfev, False

            # Try back tracking:
            if lam == 1.0:
                # First attempt, make a second order model of the cost
                # against lam.
                lam_new = - 0.5*delta_f / (cost_new - cost - delta_f)
            else:
                # Use a third order model
                r1 = (cost_new - cost - lam*delta_f)/(lam*lam)
                r2 = (cost_save - cost - lam_save*delta_f)/(lam_save*lam_save)

                a = (r1 - r2) / (lam - lam_save)
                b = (lam*r2-lam_save*r1) / (lam - lam_save)

                if a == 0:
                    lam_new = - 0.5*delta_f / b
                else:
                    d = b*b - 3*a*delta_f
                    if d < 0:
                        lam_new = 0.5*lam
                    elif b <= 0:
                        lam_new = (-b+np.sqrt(d))/(3*a)
                    else:
                        lam_new = -1 * delta_f / (b + np.sqrt(d))

                    lam_new = min(0.5*lam, lam_new)

            if np.isnan(lam_new):
                #raise ValueError("Nan in line search")
                lam_new = self._l_min*lam
                    
            lam_save = lam
            cost_save = cost_new
            lam = max(lam_new, self._l_min*lam)


def MinimizeNewton(fun, jac, hess, guess, line_search,
                  max_step=10**5, max_hev=1000, tol=1e-5):
    """Minimize a function using Newton's method with back-tracking

    Note, if Newton's method fails to find improvement a gradient descent step
    with back-tracking will be tried instead.

    Convergence is assumed when the r.m.s. jacobian is below the requested
    theshold. MinimizeNewton will guess if convergence fails due to too many 
    iterations, hessian calculations, or failure to improve the solution.

    Parameters
    ----------
    fun : function, of N variables returning a scalar
        Function to minimize
    jac : function, of N variables returning an array of N variables
        Gradient (jacobian) of the function
    hess : function, of N variables returning an array of NxN variables
        Hession of the function
    guess : array, lent N
        Initial guess for the solution
    line_search : LineSearch object
        Back-tracking line_search object
    max_step : int
        Maximum number of steps allowed
    max_hev : int
        Maximum number of Hessian evaluations allowed
    tol : float,
        Tolerance parameter for convergence. Convergence occurs when:
            np.sqrt(np.mean(jac(x)**2)) < tol
    
    Returns
    -------
    x : array,
        Best guess of the solution
    status : int,
        0 : Success
        1 : Iteration failed to improve estimate
        2 : Too many iterations
        3 : Too many hessian evaluations
    """
    need_hess = True
    nfev = 1
    nhess = 0
    x = guess
    fx = fun(x)
    for nstep in range(max_step):
        if need_hess:
            if nhess == max_hev: 
                return x, (3, nstep, nfev, nhess)
            
            j_sol = (scipy.linalg.lu_factor(hess(x)),  scipy.linalg.lu_solve)
            nhess += 1
        
        jx = jac(x)
        dx = j_sol[1](j_sol[0], -jx)

        if np.dot(jx, dx) < 0:
            x, fx, fev, failed = line_search(fun, jx, x, dx, fx, False)
            nfev += fev
        else:
            failed = True
            
        # Use gradient descent when Newton's method fails
        if failed:
            x, fx, fev, failed_descent = line_search(fun, jx, x, -jx, fx, False)
            nfev += fev


            # Try again once more with a different back tracking algorithm                
            if failed_descent:
                dx = line_search.reduce_step(-jx, x)
                alpha = 2**-4
                for _ in range(10):
                    xn = x + dx
                    fn = fun(xn)
                    nfev += 1
                    if fn < fx:
                        break
                    else:
                        dx *= alpha
                else:
                    # Neither gradient descent nor Newton's method can improve
                    # the solution
                    return x, (1, nstep, nfev, nhess)
                    
                fx = fn
                x = xn
        
        need_hess = failed or (line_search.reduction != 1.0)

        # Small enough gradient
        #if np.sqrt(np.mean(jac(x)**2)) < tol:
        #    return x, 0
        if (np.abs(jac(x))*np.abs(x)).max() < tol * max(np.abs(fx), 1):
            return x, (0, nstep, nfev, nhess)

    return x, (2, nstep, nfev, nhess)