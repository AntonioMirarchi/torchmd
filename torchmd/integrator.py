from abc import abstractmethod, ABC
import numpy as np
import torch

TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191

__all__ = [
    "Integrator",
    "LangevinIntegrator",
    "LangevinMiddleIntegrator", 
    "MTSIntegrator",
    "BAOABIntegrator",
    "get_integrator"
]


def kinetic_energy(masses, vel):
    Ekin = torch.sum(0.5 * torch.sum(vel * vel, dim=2, keepdim=True) * masses, dim=1)
    return Ekin


def maxwell_boltzmann(masses, T, replicas=1):
    natoms = len(masses)
    velocities = []
    for i in range(replicas):
        velocities.append(
            torch.sqrt(T * BOLTZMAN / masses) * torch.randn((natoms, 3)).type_as(masses)
        )

    return torch.stack(velocities, dim=0)

def kinetic_to_temp(Ekin, dof):
    return 2.0 / (dof * BOLTZMAN) * Ekin

def _first_VV(pos, vel, force, mass, dt):
    accel = force / mass
    pos += vel * dt + 0.5 * accel * dt * dt
    vel += 0.5 * dt * accel


def _second_VV(vel, force, mass, dt):
    accel = force / mass
    vel += 0.5 * dt * accel


def langevin(vel, gamma, coeff, dt, device):
    csi = torch.randn_like(vel, device=device) * coeff
    vel += -gamma * vel * dt + csi


PICOSEC2TIMEU = 1000.0 / TIMEFACTOR


class Integrator(ABC):
    def __init__(self, systems, forces, slow_forces, timestep, device, gamma=None, T=None, integrate_force=False, mts_framestep=0):
        """ 
        Base Integrator class.
        Integrator for molecular dynamics simulations.
        Parameters:
        -----------
        systems : Systems
            The systems to be simulated.
        forces : Forces
            The forces acting on the systems.
        slow_forces : Forces
            The slow forces for multiple time step integration. This is usually higher accuracy MLIP than `forces`.
        timestep : float
            The time step for the integration (in fs).
        device : torch.device
            The device to run the simulation on.
        gamma : float, optional
            The friction coefficient for Langevin dynamics (in ps^-1). If None, NVE ensemble is used.
        T : float, optional
            The temperature for Langevin dynamics (in K). If None, NVE ensemble is used.
        integrate_force : bool, optional
            If True, forces are updated directly in the integrator step.
        mts_framestep : int, optional
            If >0, enable multiple time step integration with force correction every N steps.
        -----------
        """
        self.dt = timestep / TIMEFACTOR  # Convert fs to internal units
        self.systems = systems
        self.forces = forces
        self.slow_forces = slow_forces
        self.slow_forces = slow_forces
        self.device = device
        self.T = T
        self.integrate_force = integrate_force
        self.mts_framestep = mts_framestep
        self.curl_storage = []
        
        if self.integrate_force:
            assert getattr(self.forces, 'return_forces', False), \
                "For integrate_force=True, Forces class must have return_forces=True."

        # Thermostat Parameters (Pre-calculation)
        self.gamma = None
        self.alpha = None  # Friction decay
        self.sigma = None  # Noise amplitude

        if gamma is not None and T is not None:
            # Convert gamma from ps^-1 to internal_time^-1
            # 1 ps = 1000 fs = 1000 * TIMEFACTOR internal units
            gamma = gamma / PICOSEC2TIMEU
            masses = self.forces.par.masses
            # prmtr used by Langevin (origin integrator)
            self.vcoeff = torch.sqrt(2.0 * gamma / masses * BOLTZMAN * T * self.dt).to(
                device
            )
            self.gamma = gamma
            
            # Exact Ornstein-Uhlenbeck coefficients. Used by LangevinMiddle and MTS integrators.
            # v(t+dt) = alpha * v(t) + sigma * N(0,1)
            # alpha = exp(-gamma * dt)
            self.alpha = torch.exp(torch.tensor(-self.gamma * self.dt, device=device))
            
            # sigma = sqrt( (1 - alpha^2) * kB * T / m )
            self.sigma = torch.sqrt((1.0 - self.alpha**2) * BOLTZMAN * T / masses).to(device)
        
        # MTS State
        if self.mts_framestep > 0:
            assert self.slow_forces is not None, "MTS requires slow_forces."
            self._f_correction = None
            
    def _compute_forces(self, pos, box, forces_i):
        """Helper to compute forces."""
        if self.integrate_force:
            # nnpforces class takes care of in-place update of forces_i
            # and also returns neg_dy or vec bases on the flag: integrate_neg_dy
            pot, f, curl = self.forces.compute(
                pos, box, forces_i, toNumpy=False, calculateForces=False
            )
            if isinstance(curl, torch.Tensor) and (curl != 0).any():
                self.curl_storage.append(curl)
            return pot, f
        else:
            #TOTEST
            pot, _, _, _ = self.forces.compute(pos, box, forces_i)
            return pot, self.systems.forces

    def _ou_step(self, vel):
        """Apply thermostat (Ornstein-Uhlenbeck process)."""
        if self.T is not None:
            noise = torch.randn_like(vel)
            # In-place update: v = v * alpha + noise * sigma
            vel.mul_(self.alpha).add_(noise * self.sigma)

    @abstractmethod
    def step(self, niter=1, curr_step=0):
        pass


class LangevinIntegrator(Integrator):
    """
    Standard Velocity Verlet with optional Langevin thermostat.
    
    If T is None: Pure NVE Velocity Verlet.
    If T is Set:  Velocity Verlet with Langevin thermostat.
    """
    def __init__(self, systems, forces, timestep, device, gamma=None, T=None, integrate_force=False):
        super().__init__(systems, forces, None, timestep, device, gamma, T, integrate_force)
    
    def step(self, niter=1, curr_step=0):
        s = self.systems
        masses = self.forces.par.masses

        for _ in range(niter):
            _first_VV(s.pos, s.vel, s.forces, masses, self.dt)
            
            pot, s.forces = self._compute_forces(s.pos, s.box, s.forces)
            
            if self.T is not None:
                langevin(s.vel, self.gamma, self.vcoeff, self.dt, self.device)

            _second_VV(s.vel, s.forces, masses, self.dt)

        Ekin = np.array([v.item() for v in kinetic_energy(masses, s.vel)])
        T_curr = kinetic_to_temp(Ekin, s.dof)
        return Ekin, pot.cpu().numpy(), T_curr

class LangevinMiddleIntegrator(Integrator):
    """
    Langevin Middle Integrator using the VVROR scheme. Based on discussion: https://github.com/openmm/openmm/issues/2532
    
    Sequence:
    1. VV: Full-step Kick (Velocity update)
    2. R: Full-step Drift (Position update)
    3. O: Thermostat (Ornstein-Uhlenbeck)
    4. Force Evaluation
    
    This scheme results in velocities that are effectively half-step (leapfrog),
    which often provides a more accurate representation of Kinetic Energy 
    compared to schemes that report "on-step" velocities where V and R are misaligned.
    """
    def __init__(self, systems, forces, timestep, device, gamma=None, T=None, integrate_force=False):
        super().__init__(systems, forces, None, timestep, device, gamma, T, integrate_force)
        self._initialized = False
    
    def step(self, niter=1, curr_step=0):
        s = self.systems
        masses = self.forces.par.masses
        dt = self.dt
        
        if not self._initialized:
            # Ensure forces are computed at initial positions
            if s.forces is None or torch.all(s.forces == 0.0):
                raise RuntimeError(
                    "system.forces is None. You must compute forces at initial "
                    "positions before calling integrator.step() for the first time."
                )
            
            # Shift velocity from v(0) to v(dt/2)
            # Equation: v(dt/2) = v(0) + 0.5*dt * F(x(0))/m
            accel = s.forces / masses  # a(0) = F(x(0)) / m
            s.vel += 0.5 * dt * accel  # v(0) → v(dt/2)
            
            self._initialized = True
            # State now: x(0), v(dt/2), F(x(0))
        
        for _ in range(niter):
            accel = s.forces / masses
            s.vel += dt * accel
            
            s.pos += dt * s.vel
            
            if self.T is not None:
                self._ou_step(s.vel)
            
            # This updates system.forces for next iteration
            pot, s.forces = self._compute_forces(s.pos, s.box, s.forces)

        # The velocity currently stored in s.vel is v(t + dt/2) (thermostated).
        # This is the correct velocity for accurate KE/Temperature calculation in LFMiddle.
        Ekin = np.array([v.item() for v in kinetic_energy(masses, s.vel)])
        T_curr = kinetic_to_temp(Ekin, s.dof)
        return Ekin, pot.cpu().numpy(), T_curr

class MTSIntegrator(Integrator):
    """
    Multiple Time Step (MTS) Integrator.
    Uses an Impulse-RESPA approach.
    Can also revert to NVE-MTS if T is None. This is based on https://arxiv.org/abs/2412.11569 to use slow force corrections.
    """
    def __init__(self, systems, forces, slow_forces, timestep, device, gamma=None, T=None, integrate_force=False, mts_framestep=0):
        assert T is None, "MTSIntegrator currently only supports NVE-MTS (thermostat T must be None)."
        super().__init__(systems, forces, slow_forces, timestep, device, gamma, T, integrate_force, mts_framestep)

    def step(self, niter=1, curr_step=0):
        s = self.systems
        masses = self.forces.par.masses
        dt = self.dt
        M = self.mts_framestep # e.g., 8
        
        for step in range(curr_step, curr_step + niter):
            is_mts_start = (step % M) == 0
            is_mts_end = ((step + 1) % M) == 0
            # --- 1. Slow Force Impulse (Start) ---
            # This is only executed on the first step of a slow block (k=0, M, 2M, ...)
            if is_mts_start:
                # 1a. Compute F_slow(t) (The expensive part)
                _, f_slow_curr, _ = self.slow_forces.compute(
                    s.pos, s.box, s.forces, toNumpy=False, calculateForces=False
                )
                
                # 1b. Calculate Correction F_slow - F_fast 
                # (s.forces holds F_fast from the *previous* step, which is F_fast(t-dt_slow) if at step 0)
                self._f_correction = f_slow_curr - s.forces 
                
                # 1c. Apply the first half of the slow force impulse (V_slow^1/2)
                # Use the full slow time step M*dt
                s.vel += 0.5 * (dt * M) * (self._f_correction / masses)

            # --- 2. Fast Inner Loop (Velocity Verlet V_fast R V_fast) ---
            # Applied M times over the slow step duration.
            
            # 2a. Half Kick Fast (V_fast^1/2)
            s.vel += 0.5 * dt * (s.forces / masses)
            
            # 2b. Drift Fast (R)
            s.pos += s.vel * dt
            
            # 2c. Force Update Fast (Compute F_fast(t+dt))
            pot, s.forces = self._compute_forces(s.pos, s.box, s.forces)

            # 2d. Half Kick Fast (V_fast^1/2)
            s.vel += 0.5 * dt * (s.forces / masses)
            
            # --- 3. Slow Force Impulse (End) ---
            # This is only executed on the last step of a slow block (k=M-1, 2M-1, ...)
            if is_mts_end:
                # 3a. Apply the second half of the slow force impulse (V_slow^1/2)
                # Use the full slow time step M*dt
                s.vel += 0.5 * (dt * M) * (self._f_correction / masses)
                
                # 3b. Clear correction for next slow step
                self._f_correction = None
                        
        Ekin = np.array([v.item() for v in kinetic_energy(masses, s.vel)])
        T_curr = kinetic_to_temp(Ekin, s.dof)
        return Ekin, pot.cpu().numpy(), T_curr


class BAOABIntegrator(Integrator):
    """
    Explicit BAOAB Integrator.
    """
    def __init__(self, systems, forces, timestep, device, gamma=None, T=None, integrate_force=False):
        super().__init__(systems, forces, None, timestep, device, gamma, T, integrate_force)
    
    def step(self, niter=1, curr_step=0):
        s = self.systems
        masses = self.forces.par.masses
        dt = self.dt
        
        for _ in range(niter):
            # B: First half kick
            s.vel += 0.5 * dt * (s.forces / masses)
            
            # A: First half drift
            s.pos += 0.5 * dt * s.vel
            
            # O: Thermostat
            self._ou_step(s.vel)
            
            # A: Second half drift
            s.pos += 0.5 * dt * s.vel
            
            # Force Update
            pot, s.forces = self._compute_forces(s.pos, s.box, s.forces)
            
            # B: Second half kick
            s.vel += 0.5 * dt * (s.forces / masses)
        
        Ekin = np.array([v.item() for v in kinetic_energy(masses, s.vel)])
        T_curr = kinetic_to_temp(Ekin, s.dof)
        return Ekin, pot.cpu().numpy(), T_curr
