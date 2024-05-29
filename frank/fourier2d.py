
import numpy as np

class DiscreteFourierTransform2D(object):
    def __init__(self, Rmax, N, nu=0):
        # Remember that now N is to create N**2 points in image plane.
        self.Xmax  = 2*Rmax # [Rmax] = rad.
        self.Ymax = 2*Rmax
        self.Nx = N 
        self.Ny = N
        self.N = N**2 # Number of points we want to use in the 2D-DFT.

        # Real space collocation points
        x1n = np.linspace(0, self.Xmax, self.Nx) # rad
        x2n = np.linspace(0, self.Ymax, self.Ny) # rad
        x1n, x2n = np.meshgrid(x1n, x1n, indexing='ij')
        # x1n.shape = N**2 X 1, so now, we have N**2 collocation points in the image plane.
        x1n, x2n = x1n.reshape(-1), x2n.reshape(-1) # x1n = x_array and x2n = y_array
        dx = 2*self.Xmax/self.N
        dy = 2*self.Ymax/self.N

        # Frequency space collocation points.
        # The [1:] is because to not consider the 0 baseline. But we're missing points. 
        q1n = np.fft.fftfreq(self.Nx+1, d = dx)[1:]
        q2n = np.fft.fftfreq(self.Ny+1, d = dy)[1:]
        q1n, q2n = np.meshgrid(q1n, q2n, indexing='ij') 
        # q1n.shape = N**2 X 1, so now, we have N**2 collocation points.
        q1n, q2n = q1n.reshape(-1), q2n.reshape(-1) # q1n = u_array and q2n = v_array

        self.Xn = x1n
        self.Yn = x2n
        self.Un = q1n
        self.Vn = q2n

    def get_collocation_points(self):        
        return np.array([self.Xn, self.Yn]), np.array([self.Un, self.Vn])

    def coefficients(self, u = None, v = None, direction="forward"):
        if direction == 'forward':
            norm = 1
            factor = -2j*np.pi/self.Nx
        elif direction == 'backward':
            norm = 1 / self.N
            factor = 2j*np.pi/self.Nx
        else:
            raise AttributeError("direction must be one of {}"
                                 "".format(['forward', 'backward']))
        if u is None:
            u = self.Un
            v = self.Vn
        
        if direction == "forward":
            H = norm * np.exp(factor*(np.outer(u, self.Xn) + np.outer(v, self.Yn)))
        else:
            H = norm * np.exp(factor*(np.outer(u, self.Xn) + np.outer(v, self.Yn)))
        return H


    def transform(self, f, u, v, direction="forward"):
        Y = self.coefficients(u, v, direction=direction)
        return np.dot(Y, f)
          
          
    @property
    def size(self):
        """Number of points used in the 2D-DFT"""
        return self.N
    
    @property
    def uv_points(self):
        """u and v  collocation points"""
        return self.Un, self.Vn

    @property
    def q(self):
        """Frequency points"""
        return np.hypot(self.Un, self.Vn)