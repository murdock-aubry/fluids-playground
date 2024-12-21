import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.sparse import eye
import scipy.sparse
from numerics import construct_sparse_matrix



class Stable2dObject():

    def __init__(
            self, 
            ngrid = 64, 
            diffusion = 0.0001, 
            viscosity = 0.001, 
            dt = 0.1, 
            dye_diffusion=0.00005,
            noslip_bdy = True,
            object_mask = None,
            force = 0.0,
        ):

        self.ngrid = ngrid
        self.dt = dt
        self.diffusion = diffusion
        self.viscosity = viscosity
        self.dye_diffusion = dye_diffusion
        self.noslip_bdy = noslip_bdy
        self.object_mask = object_mask
        self.force = force 

        self.velocity = np.zeros((2, self.ngrid, self.ngrid))
        self.dye = np.zeros((self.ngrid, self.ngrid))



    def forward(self):
        u, v = self.velocity[0], self.velocity[1]

        if self.noslip_bdy:
            u, v = self.apply_no_slip_boundary(u, v)

        if self.object_mask is not None:
            u[self.object_mask == 1] = 0
            v[self.object_mask == 1] = 0

        # Apply forces before the regular steps
        u, v = self.apply_forces(u, v)
        
        # Rest of the original forward method
        if self.noslip_bdy:
            u, v = self.apply_no_slip_boundary(u, v)

        u = self.diffuse(u, self.diffusion, self.dt)
        v = self.diffuse(v, self.diffusion, self.dt)
        u, v = self.project(u, v, self.dt)
        u = self.advect(u, u, v, self.dt)
        v = self.advect(v, u, v, self.dt)

        u, v = self.project(u, v, self.dt)
        
        if self.noslip_bdy:
            u, v = self.apply_no_slip_boundary(u, v)
        if self.object_mask is not None:
            u[self.object_mask == 1] = 0
            v[self.object_mask == 1] = 0
        
        self.velocity[0] = u
        self.velocity[1] = v

        # Update dye field
        self.dye = self.diffuse(self.dye, self.dye_diffusion, self.dt)
        self.dye = self.advect(self.dye, u, v, self.dt)
        self.dye = np.clip(self.dye, 0, 100)
        self.dye[self.object_mask == 1] = 0

        return self.velocity, self.dye


    def diffuse(self, field, diffusion, dt, tolerance=1e-6, max_iterations=1000):
        N = field.shape[0]
        alpha = dt * diffusion * (N - 2) ** 2

        # Directly modify field in-place to save memory
        for iteration in range(max_iterations):
            # Efficiently calculate the neighbors using slicing
            val = (field[:-2, 1:-1] + field[2:, 1:-1] +
                field[1:-1, :-2] + field[1:-1, 2:])
            
            # Update the field in-place without copying
            field[1:-1, 1:-1] = (field[1:-1, 1:-1] + alpha * val) / (1 + 4 * alpha)

            # Check for convergence
            max_change = np.abs(field - field.copy()).max()  # `field.copy()` to avoid in-place comparison
            
            # Early exit if convergence is reached
            if max_change < tolerance:
                break

        return field

    def project(self, u, v, dt=0.1):

        h, w = u.shape

        # Compute divergence on interior grid
        div = np.zeros((h-2, w-2))
        div = 0.5 * (u[2:, 1:-1] - u[:-2, 1:-1] + v[1:-1, 2:] - v[1:-1, :-2])

        A = construct_sparse_matrix(div, u.shape)

        # Solve Poisson equation
        p = spsolve(A, div.flatten()).reshape(div.shape)
        p_padded = np.pad(p, ((1, 1), (1, 1)), mode='constant')

        # Apply pressure gradient
        u[1:-1, 1:-1] -= 0.5 * dt * (p_padded[1:-1, 2:] - p_padded[1:-1, 1:-1])
        v[1:-1, 1:-1] -= 0.5 * dt * (p_padded[2:, 1:-1] - p_padded[1:-1, 1:-1])

        return u, v

    def advect(self, field, u, v, dt):
        N = field.shape[0]

        x = np.arange(N)[:, None] - u * dt  
        y = np.arange(N)[None, :] - v * dt 

        # Clamp values to ensure they stay within bounds
        x = np.clip(x, 0.5, N - 1.5)
        y = np.clip(y, 0.5, N - 1.5)

        # Find the integer parts (i0, j0) and fractional parts (sx, sy)
        i0, j0 = np.floor(x).astype(int), np.floor(y).astype(int)
        i1, j1 = i0 + 1, j0 + 1
        sx, sy = x - i0, y - j0

        # Perform bilinear interpolation
        new_field = (
            (1 - sx) * (1 - sy) * field[i0, j0]
            + sx * (1 - sy) * field[i1, j0]
            + (1 - sx) * sy * field[i0, j1]
            + sx * sy * field[i1, j1]
        )

        return new_field


    def apply_no_slip_boundary(self, u, v):
        # Assuming u and v are velocity components in x and y directions respectively
        # u[0, :] = 0  # Top
        # u[-1, :] = 0 # Bottom
        # u[:, 0] = 0  
        # u[:, -1] = 0 
        
        # v[0, :] = 0
        # v[-1, :] = 0
        # v[:, 0] = 0
        # v[:, -1] = v[:, 0]

        return u, v


    def apply_forces(self, u, v):
        ngrid = self.ngrid
        
        v += self.force * self.dt
        return u, v
