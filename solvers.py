import numpy as np
import torch
import matplotlib.pyplot as plt

class StochasticNoise:
    """
    generates spatially correlated gaussian noise for stochastic PDE forcing

    the noise field has a prescribed spatial correlation structure defined by its fourier-space energy spectrum. 
    for colored noise with correlation length l_c, the amplitude profile is:
        - amplitude(k) ~ sigma * exp(-0.5 * (k * l_c)^2)

    short correlation length = rough/spiky forcing
    long correlation length = smooth/large-scale forcing
    """

    def __init__(self, sigma: float, correlation_length: float, seed: int = None):
        self.sigma = sigma                              # noise intensity
        self.correlation_length = correlation_length    # spatial correlation scale (meters)
        self.seed = seed
        self.rng = torch.Generator()
        if seed is not None: 
            self.rng.manual_seed(seed)
        
    def sample_1d(self, n_samples: int, n_timesteps:int, Nx:int, dx: float):
        """
        generates noise realizations for 1D problems

        each realization is a space-time field xi(x, t) with prescribed spatial correlation and white-in-time structure
        
        returns: 
            noise_field: (n_samples, n_timesteps, Nx) tensor of forcing values
        """

        # wavenumber for FFT
        k = torch.fft.fftfreq(Nx, d=dx) * 2 * np.pi     # rad/m
        k_mag = torch.abs(k)

        # spectral amplitude profile (gaussian correlation -> gaussian spectrum)
        lc = self.correlation_length
        amplitude = self.sigma * torch.exp(-0.5 * (k_mag * lc)**2)
        amplitude[0] = 0.0          # zero the mean mode - no DC component

        # sample random fourier coefficients and scale by amplitude
        real = torch.randn(n_samples, n_timesteps, Nx, generator=self.rng)
        imag = torch.randn(n_samples, n_timesteps, Nx, generator=self.rng)
        noise_hat = torch.complex(real, imag) * amplitude.unsqueeze(0).unsqueeze(0)

        # inverse FFT -> physical space noise field 
        noise_field = torch.fft.ifft(noise_hat, dim = 1).real

        # renormalize so variance matches sigma^2
        noise_field = noise_field / (noise_field.std() + 1e-8) * self.sigma
 
        return noise_field
    
    def sample_2d(self, n_samples: int, n_timesteps: int, Nx: int, Ny: int, dx: float, dy: float):
        """
        noise relization for 2D problems

        returns:
            noise_field: (n_samples, n_timesteps, Nx, Ny) tensor
        """

        # 2D wavenumber grid
        kx = torch.fft.fftfreq(Nx, d = dx) * 2 * np.pi
        ky = torch.fft.fftfreq(Nx, d = dy) * 2 * np.pi
        kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing='ij')
        k_mag = torch.sqrt(kx_grid**2 + ky_grid**2)

        # spectral amplitude 
        lc = self.correlation_length
        amplitude = self.sigma * torch.exp(-0.5 * (k_mag * lc)**2)
        amplitude[0, 0] = 0.0  #zero mean

        #random fourier coefficients
        real = torch.randn(n_samples, n_timesteps, Nx, Ny, generator=self.rng)
        imag = torch.randn(n_samples, n_timesteps, Nx, Ny, generator=self.rng)
        amp = amplitude.unsqueeze(0).unsqueeze(0)
        noise_hat = torch.complex(real, imag) * amp
 
        noise_field = torch.fft.ifft2(noise_hat, dim=(-2, -1)).real
        noise_field = noise_field / (noise_field.std() + 1e-8) * self.sigma
 
        return noise_field
    

class StochasticBurgers1D:
    """
    Pseudo-spectral for 1D stochastic burgers equation

    du/dt + u * du/dx = nu * d²u/dx² + sigma * dW/dt

    Similar to 1D navier-stokes;
    - same nonlinear convection and viscous diffusion
    - drops pressure and incompressibility
    
    time integration using (IMEX - implicit-explicit)
    """

    def __init__(self, lx: float, nx: int, t: float, nt: float, nu: float):
        """
        lx - domain length (meters), periodic on [0, Lx)
        nx - number of grid points
        t - final time (s)
        nt - number of time steps
        nu - kinematic viscosity (m^2/s)
        """

        self.lx = lx
        self.nx = nx
        self.t = t
        self.nt = nt
        self.nu = nu
 
        self.dx = lx / nx
        self.dt = t / nt

        # spatial grid
        self.x = torch.linspace(0, lx * (1 - 1/nx), nx, dtype=torch.float64)

        # wavenumbers
        self.k = torch.fft.fftfreq(nx, d=self.dx).to(torch.float64) * 2 * np.pi    #rad/m)

        self.k_sq = self.k**2 
        self.ik = 1j * self.k   # i*k for spectral derivatives (in fourier space, derivatives become multiplication)
                                # 1j indicates imaginary i in complex number

        # dealiasing mask (2/3)
        # used to prevent nonlinear aliasing error from u^2 term
        self.dealis_mask = torch.ones(nx, dytpe = torch.float64)
        k_max = nx // 3
        self.dealis_mask[k_max:-k_max] = 0.0

        # integrating factor for diffusion
        self.decay = torch.exp(self.nu * self.k_sq * self.dt)

    def _nonlinear_term(self, u_hat):
        """
        computes -u * du/dx in fourier space

        conservative form: -0.5 * d(u^2)/dx
        fourier space: -0.5 * ik * FFT(u^2)

        this nonlinear term causes mode interactions in Fourier space 
            - mode interactions: transfer of energy between different spatial frequencies
        at high frequencies, interactions produce frequenceis too large for discrete grid representation
        these unresolved frequencies are foldered back into lower ones, which create aliasing errors

        spectral methods use 2/3 rule, which reduce such aliasing errors
    
        dealising masks zeros high-frequency modes before computing the product in physical space
            - this prevents spurious energy transfer
        """


        # Re [Dealised Fourier components in Fourier space, then transformed back to physical space]
        u = torch.fft.ifft(u_hat * self.dealias_mask).real 
        u_sq_hat = torch.fft.fft(u**2)                          # nonlinear product back to fourier
        return -0.5 * self.ik * u_sq_hat
    
    def generate_initial_conditions(self, n_samples: int, seed: int = None):
        """
        generate smooth random intial conditions using low-pass filtered random fourier coefficients
        
        random initial condition are used to probe how the system behaves across many possible states
            - avoids bias from special initial conditions
            - sample from all smooth velocity field space

        only the lowest 1/4 of modes are populated  
            - this guarantees smooth fields that physically reasonable for viscous flow
        """

        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        
        n_active = self.Nx // 4         # only populate low-freq modes
        real = torch.randn(n_samples, self.Nx, generator=rng, dtype=torch.float64)
        imag = torch.randn(n_samples, self.Nx, generator=rng, dtype=torch.float64)
        u0_hat = torch.complex(real, imag)

        # zero high-frequency modes for smoothness
        u0_hat[:, n_active: -n_active] = 0
        u0 = torch.fft.ifft(u0_hat, dim=1).real

        return u0
    
    def solve(self, u0, noise_field):
        """
        solves stochastic burgers equation from initial condition u0 using IMEX-Huen scheme

        IMEX-Huen:
            predictor: u* = decay * u + dt * decay * NL(u) + noise * sqrt(dt)
            corrector: u^{n+1} = decay * u + 0.5*dt*(decay*NL(u) +NL(u*)) + noise*sqrt(dt)

        Args: 
            u0: inital condition (Nx,) tensor
            noise_field: stochastic forcing (Nt, Nx) tensor
        
        evolves the stochastic Burgers equation forward in time using:
            - FFT/spectral representation in space
            - Heun's method in time for nonlinear part 
            - integrating factor for diffusion
            - additive stochastic forcing
            
        returns: 
            u_final: solution at time T, shape (Nx,)
        """
        
        u_hat = torch.fft.fft(u0)       # converts IC from physical to fourier space 

        # time loop
        for n in range(self.Nt):
            noise_hat = torch.fft.fft(noise_field[n].to(torch.complex128))      # transform noise into Fourier space

            # heun predictor
            nl1 = self._nonlinear_term(u_hat)
            u_hat_pred = self.decay * u_hat + self.dt * self.decay * nl1 + noise_hat * np.sqrt(self.dt)

            # heun corrector
            nl2 = self._nonlinear_term(u_hat_pred)
            u_hat = self.decay * u_hat + 0.5 * self.dt * (self.decay * nl1 + nl2) + noise_hat * np.sqrt(self.dt)
        
        u_final = torch.fft.ifft(u_hat).real.float()
        return u_final
    
    def solve_batch(self, u0_batch, noise_batch):
        """solve for a batch of pairs (initial conditions, noise)"""
        solutions = []
        for i in range(u0_batch.shape[0]):
            sol = self.solve(u0_batch[i], noise_batch[i])
            solutions.append(sol)
        return torch.stack(solutions)
    


class StochasticNavierStokes2D: 
    """
    pseudo-spectral solver for 2D stochastic navier-stokes equations in voriticity-streamfunction form 

    take the curl of the 2D momentum equation:
        - eliminates pressure entirely
        - reuduces to a single scalar equation for vorticity

    d(omega)/dt = J(psi, omega) = nu * Lap(omega) + curf(f) 

    omega = dv/dx - du/dy           (scalar vorticity in 2D)
    psi = streamfunction            (defined by Lap(psi) = -omega)

    Jacobian/advection
    J(psi, omega) = d(psi)/dx * d(omega)/dy - d(psi)/dy * d(omega)/dx

    velocity recovered from streamfunction
    u = d(psi)dy
    v = -d(psi)/dx     

    this form was chosen for convenience: 
        - pressure is gone, meaning no poisson equation to solve at each step
        - incompressibility (div(u) = 0) is automatically satisfied
        - scalar equation in 2D (voriticity is a scalar, not vector)

    time integration: IMEX with crank-nicolson (diffusion) + adams-bashforth (advection)
    """

    def __init__(self, lx: float, ly: float, nx: int, ny: int, t: float, nt: int, nu: float):
        """
        Args: 
            Lx, Ly: domain size in x and y (meters), periodic on [0, L)
            Nx, Ny: grid points in each direction
            T: final time (seconds)
            Nt: number of time steps
            nu: kinematic viscosity (m^2/s), related to Reynolds number by Re = U*L/nu
        """

        self.lx = lx
        self.ly = ly
        self.nx = nx
        self.ny = ny
        self.t = t
        self.nt = nt
        self.nu = nu

        self.dx = lx/nx
        self.dy = ly/ny
        self.dt = t/nt

        # spatial grids
        self.x = torch.linspace(0, lx * (1 - 1/nx), nx, dtype=torch.float64)
        self.y = torch.linspace(0, ly * (1 - 1/ny), ny, dtype=torch.float64)
 
        #wavenumber grids
        self.kx = torch.fft.fftfreq(nx, d=self.dx).to(torch.float64) * 2 * np.pi
        self.ky = torch.fft.fftfreq(ny, d=self.dy).to(torch.float64) * 2 * np.pi
        self.kx_grid, self.ky_grid = torch.meshgrid(self.kx, self.ky, indexing='ij')

        # |k|^2 and its inverse (for poisson solve: psi_hat = -omega_hat / |k|^2)
        self.k_sq = self.kx_grid**2 + self.ky_grid**2
        self.k_sq_inv = torch.where(             #avoid division by zero at k=0
            self.k_sq > 0, 1.0 / self.k_sq, torch.zeros_like(self.k_sq))
 
        self.ikx = 1j * self.kx_grid       #i*kx for x-derivatives
        self.iky = 1j * self.ky_grid        #i*ky for y-derivatives

        #dealiasing mask (2/3 rule in each direction)
        mask_x = torch.ones(nx, dtype=torch.float64)
        mask_y = torch.ones(ny, dtype=torch.float64)
        mask_x[nx // 3 : -nx // 3] = 0.0
        mask_y[ny // 3 : -ny // 3] = 0.0
        self.dealias_mask = mask_x.unsqueeze(1) * mask_y.unsqueeze(0)

        #crank-nicolson coefficients for diffusion
        # (1 + 0.5*dt*nu*k^2) * w^{n+1} = (1 - 0.5*dt*nu*k^2) * w^n + RHS
        self.cn_implicit = 1.0 + 0.5 * self.dt * self.nu * self.k_sq
        self.cn_explicit = 1.0 - 0.5 * self.dt * self.nu * self.k_sq
        self.cn_implicit_inv = 1.0 / self.cn_implicit

    def _omega_to_velocity(self, omega_hat):
        """
        recover velocity field from voriticity via streamfunction

        solve poisson equation for streamfunction
            Lap(psi) = -omega  =>  -|k|^2 * psi_hat = -omega_hat
            psi_hat = omega_hat / |k|^2
        
        take curl of streamfunction to get velocity
            u = dpsi/dy  =>  u_hat = i*ky * psi_hat
            v = -dpsi/dx =>  v_hat = -i*kx * psi_hat
        """
        psi_hat = -omega_hat * self.k_sq_inv
        u_hat = self.iky * psi_hat                   #dpsi/dy
        v_hat = -self.ikx * psi_hat                  #-dpsi/dx
        return u_hat, v_hat
    
    def _nonlinear_term(self, omega_hat):
        """
        compute advection: -J(psi, omega) = -(u * domega/dx + v * domega/dy)

        Computed in physical space with dealiasing:
            1. get velocity (u, v) from vorticity
            2. get vorticity gradients (domega/dx, domega/dy)
            3. multiply in physical space
            4. FFT back
        """
        # velocity from voriticity: 
        u_hat, v_hat = self._omega_to_velocity(omega_hat)

        # voritcity gradients in fourier space
        dwdx_hat = self.ikx * omega_hat
        dwdy_hat = self.iky * omega_hat

        #transform to physical space (dealiased)
        u = torch.fft.ifft2(u_hat * self.dealias_mask).real
        v = torch.fft.ifft2(v_hat * self.dealias_mask).real
        dwdx = torch.fft.ifft2(dwdx_hat * self.dealias_mask).real
        dwdy = torch.fft.ifft2(dwdy_hat * self.dealias_mask).real
 
        #advection in physical space, then back to fourier
        advection = u * dwdx + v * dwdy
        return -torch.fft.fft2(advection)
    
    def generate_initial_condition(self, n_samples: int, seed: int = None):
        """
        generate smooth random vorticity fields
 
        uses a prescribed energy spectrum: E(k) ~ k^4 * exp(-(k/k0)^2)
            - produces smooth, vortex-dominated initial conditions
            - k^4 factor ensures smoothness at large scales
            - exponential cutoff removes small-scale noise
        """
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
 
        k_mag = torch.sqrt(self.k_sq)
        k0 = 4.0   # peak wavenumber (sets dominant vortex size)
 
        # energy spectrum shape
        energy_profile = k_mag**4 * torch.exp(-(k_mag / k0)**2)
        amplitude = torch.sqrt(energy_profile / (k_mag + 1e-10))
 
        omega0_batch = []
        for _ in range(n_samples):
            real = torch.randn(self.nx, self.ny, generator=rng, dtype=torch.float64)
            imag = torch.randn(self.nx, self.ny, generator=rng, dtype=torch.float64)
            omega_hat = torch.complex(real, imag) * amplitude
            omega_hat[0, 0] = 0                         # enforce zero mean vorticity
            omega = torch.fft.ifft2(omega_hat).real
            omega0_batch.append(omega)
 
        return torch.stack(omega0_batch)
 
    def solve(self, omega0, noise_field):
        """
        solves stochastic vorticity equation
 
        time integration: crank-nicolson + adams-bashforth 2
            first step uses forward euler for advection (no previous nonlinear term (NL) available)
            subsequent steps use AB2: 1.5*NL^n - 0.5*NL^{n-1}
        Args:
            omega0: initial vorticity field (nx, ny) tensor
            noise_field: stochastic forcing (nt, nx, ny) tensor
 
        Returns:
            omega_final: vorticity at time t, shape (nx, ny)
        """
        omega_hat = torch.fft.fft2(omega0)
        nl_prev = None      # stores NL from previous step (for adams-bashforth)
 
        for n in range(self.nt):
            noise_hat = torch.fft.fft2(noise_field[n].to(torch.complex128))
 
            # nonlinear advection term
            nl = self._nonlinear_term(omega_hat)
 
            if nl_prev is None:
                # first step: forward euler for advection
                rhs = (self.cn_explicit * omega_hat
                       + self.dt * nl
                       + noise_hat * np.sqrt(self.dt))
            else:
                # adams-bashforth 2 for advection: 1.5*NL^n - 0.5*NL^{n-1}
                rhs = (self.cn_explicit * omega_hat + self.dt * (1.5 * nl - 0.5 * nl_prev) + noise_hat * np.sqrt(self.dt))
 
            # implicit solve for diffusion, which is just a division in fourier space 
            omega_hat = self.cn_implicit_inv * rhs
            nl_prev = nl
 
        omega_final = torch.fft.ifft2(omega_hat).real.float()
        return omega_final
 
    def solve_batch(self, omega0_batch, noise_batch):
        """solve for a batch of pairs (initial conditions, noise)"""
        solutions = []
        for i in range(omega0_batch.shape[0]):
            sol = self.solve(omega0_batch[i], noise_batch[i])
            solutions.append(sol)
        return torch.stack(solutions)
 
    def compute_energy_spectrum(self, omega):
        """
        compute shell-averaged kinetic energy spectrum E(k)
        
        E(k) = 0.5 * sum_{|k'| in [k-0.5, k+0.5]} (|u_hat|^2 + |v_hat|^2)
        """
        omega_hat = torch.fft.fft2(omega.to(torch.float64))
        u_hat, v_hat = self._omega_to_velocity(omega_hat)
 
        energy_hat = 0.5 * (torch.abs(u_hat)**2 + torch.abs(v_hat)**2)
 
        k_mag = torch.sqrt(self.k_sq)
        k_max = int(min(self.nx, self.ny) / 2)
        spectrum = torch.zeros(k_max)
 
        for ki in range(k_max):
            mask = (k_mag >= ki - 0.5) & (k_mag < ki + 0.5)
            spectrum[ki] = energy_hat[mask].sum()
 
        return spectrum
 
 
class DataGenerator:
    """
    orchestrates solver + noise to produce training datasets
 
    generates paired (input, output) data:
        input: initial condition + noise representation + spatial coordinates
        output: solution at final time
 
    the neural operator learns the map: (IC, noise) -> solution
    """
 
    def __init__(self, solver, noise_gen: StochasticNoise):
        self.solver = solver
        self.noise_gen = noise_gen
 
    def generate(self, n_samples: int, seed: int = 42, verbose: bool = True):
        """
        generate dataset of SPDE solution pairs
 
        returns dict with:
            'u0': initial conditions
            'noise': time-averaged noise fields (spatial input to model)
            'solutions': solutions at final time
        """
        if verbose:
            print(f"generating {n_samples} samples...")
 
        # initial conditions
        u0_batch = self.solver.generate_initial_condition(n_samples, seed=seed)
        if verbose:
            print(f"  initial conditions: {u0_batch.shape}")
 
        # noise realizations
        is_2d = hasattr(self.solver, 'ny')
        if is_2d:
            noise_fields = self.noise_gen.sample_2d(
                n_samples, self.solver.nt,
                self.solver.nx, self.solver.ny,
                self.solver.dx, self.solver.dy
            )
        else:
            noise_fields = self.noise_gen.sample_1d(n_samples, self.solver.nt, self.solver.nx, self.solver.dx)
        if verbose:
            print(f"  noise fields: {noise_fields.shape}")
 
        # solve each realization
        solutions = []
        for i in range(n_samples):
            sol = self.solver.solve(u0_batch[i], noise_fields[i])
            solutions.append(sol)
            if verbose and (i + 1) % max(1, n_samples // 5) == 0:
                print(f"  solved {i+1}/{n_samples}")
 
        solutions = torch.stack(solutions)
 
        # time-averaged noise field: used as spatial input channel to the model
        #   - collapses the temporal dimension while preserving spatial structure
        noise_spatial = noise_fields.mean(dim=1).float()
 
        return {
            'u0': u0_batch.float(),
            'noise': noise_spatial,
            'solutions': solutions,
        }
 
    def generate_splits(self, n_train: int, n_val: int, n_test: int, verbose=True):
        """generate train/val/test splits with different seeds"""
        train = self.generate(n_train, seed=42, verbose=verbose)
        val = self.generate(n_val, seed=43, verbose=verbose)
        test = self.generate(n_test, seed=44, verbose=verbose)
        return train, val, test
 
 
if __name__ == "__main__":
    """=== test ==="""
 
    print("=== 1D stochastic burgers test ===")
    solver_1d = StochasticBurgers1D(
        lx=2*np.pi, nx=64, t=0.5, nt=200, nu=0.01
    )
    noise_1d = StochasticNoise(sigma=0.05, correlation_length=0.3, seed=42)
 
    u0 = solver_1d.generate_initial_condition(3, seed=0)
    noise_field = noise_1d.sample_1d(3, solver_1d.nt, solver_1d.nx, solver_1d.dx)
 
    for i in range(3):
        sol = solver_1d.solve(u0[i], noise_field[i])
        print(f"  sample {i}: u0 range [{u0[i].min():.3f}, {u0[i].max():.3f}] "
              f"-> sol range [{sol.min():.3f}, {sol.max():.3f}]")
 
    print("\n=== 2D stochastic NS test ===")
    solver_2d = StochasticNavierStokes2D(
        lx=2*np.pi, ly=2*np.pi, nx=32, ny=32,
        t=0.5, nt=100, nu=1e-3
    )
    noise_2d = StochasticNoise(sigma=0.05, correlation_length=0.5, seed=42)
 
    omega0 = solver_2d.generate_initial_condition(2, seed=0)
    noise_field_2d = noise_2d.sample_2d(2, solver_2d.nt, 32, 32, solver_2d.dx, solver_2d.dy)
 
    for i in range(2):
        sol = solver_2d.solve(omega0[i], noise_field_2d[i])
        print(f"  sample {i}: omega0 range [{omega0[i].min():.3f}, {omega0[i].max():.3f}] "
              f"-> sol range [{sol.min():.3f}, {sol.max():.3f}]")
 
    print("\ndone.")
 