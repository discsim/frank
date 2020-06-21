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
    armijo : float.

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
                  max_step=10**5, max_jev=1000, tol=1e-5):
    """Minimize a function using Newton's method with back-tracking

    Note, if Newton's method fails to find improvement a gradient descent step
    with back-tracking will be tried instead.

    Convergence is assumed when the r.m.s. jacobian is below the requested
    theshold. MinimizeNewton will guess if convergence fails due to too many 
    iterations, jacobian calculations, or failure to improve the solution.

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
    mac_jev : int 
        Maximum number of jacobian evaluations allowed
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
        3 : Too many jacobian evaluations
    """
    need_jac = True
    njac = 0
    x = guess
    fx = fun(x)
    for _ in range(max_step):
        if need_jac:
            if njac == max_jev: 
                return x, 3
            
            j_sol = (scipy.linalg.lu_factor(hess(x)),  scipy.linalg.lu_solve)
            njac += 1
        
        jx = jac(x)
        dx = j_sol[1](j_sol[0], -jx)

        if np.dot(jx, dx) < 0:
            x, fx, _, failed = line_search(fun, jx, x, dx, fx, False)
        else:
            failed = True
            
        # Use gradient descent when Newton's method fails
        if failed:
            x, fx, _, failed_descent = line_search(fun, jx, x, -jx, fx, False)

            # Try again once more with a different back tracking algorithm                
            if failed_descent:
                dx = line_search.reduce_step(-jx, x)
                alpha = 2**-4
                for _ in range(10):
                    xn = x + dx
                    fn = fun(xn)
                    if fn < fx:
                        break
                    else:
                        dx *= alpha
                else:
                    # Neither gradient descent nor Newton's method can improve
                    # the solution
                    return x, 1
                    
                fx = fn
                x = xn
        
        need_jac = failed or (line_search.reduction != 1.0)

        # Small enough gradient
        if np.sqrt(np.mean(jac(x)**2)) < tol:
            return x, 0

    return x, 2