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
            
    def _compute_forces(self, pos, forces_prev):
        """Helper to compute forces."""
        if self.integrate_force:
            pot, vec, curl = self.forces.compute(
                pos, self.systems.box, forces_prev, toNumpy=False, calculateForces=False
            )
            if isinstance(curl, torch.Tensor) and (curl != 0).any():
                self.curl_storage.append(curl)
            return pot, vec
        else:
            pot, _ = self.forces.compute(pos, self.systems.box, forces_prev)
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
    If T is Set:  Velocity Verlet with BAOAB-style thermostatting.
    """
    def __init__(self, systems, forces, timestep, device, gamma=None, T=None, integrate_force=False):
        super().__init__(systems, forces, None, timestep, device, gamma, T, integrate_force)
    
    def step(self, niter=1, curr_step=0):
        s = self.systems
        masses = self.forces.par.masses
        dt = self.dt
        
        for _ in range(niter):
            _first_VV(s.pos, s.vel, s.forces, masses, self.dt)
            pot, s.forces = self._compute_forces(s.pos, s.forces)
            
            if self.T is not None:
                langevin(s.vel, self.gamma, self.vcoeff, self.dt, self.device)

            _second_VV(s.vel, s.forces, masses, self.dt)

        Ekin = np.array([v.item() for v in kinetic_energy(masses, s.vel)])
        T_curr = kinetic_to_temp(Ekin, self.systems.dof)
        return Ekin, pot.cpu().numpy(), T_curr

class LangevinMiddleIntegrator(Integrator):
    """
    Langevin Middle Integrator using the VVROR scheme.
    
    Sequence:
    1. V: Half-step Kick
    2. R: Full-step Drift (Position update)
    3. Force Update
    4. O: Thermostat (Ornstein-Uhlenbeck)
    5. V: Half-step Kick
    
    This scheme results in velocities that are effectively half-step (leapfrog),
    which often provides a more accurate representation of Kinetic Energy 
    compared to schemes that report "on-step" velocities where V and R are misaligned.
    """
    def __init__(self, systems, forces, timestep, device, gamma=None, T=None, integrate_force=False):
        super().__init__(systems, forces, None, timestep, device, gamma, T, integrate_force)
    
    def step(self, niter=1, curr_step=0):
        s = self.systems
        masses = self.forces.par.masses
        dt = self.dt
        
        for _ in range(niter):
            # --- V: First Half Kick ---
            # v(t + dt/2) = v(t) + 0.5 * a(t) * dt
            accel = s.forces / masses
            s.vel += 0.5 * dt * accel
            
            # --- R: Full Position Update (Drift) ---
            # r(t + dt) = r(t) + v(t + dt/2) * dt
            s.pos += s.vel * dt
            
            # --- Force Update ---
            # F(t + dt) computed at new positions
            pot, s.forces = self._compute_forces(s.pos, s.forces)
            
            # --- O: Thermostat ---
            # Applied after force evaluation but before the second kick.
            # If T is None, this step is skipped (NVE limit).
            if self.T is not None:
                self._ou_step(s.vel)
            
            # --- V: Second Half Kick ---
            # v(t + dt) = v_thermo + 0.5 * a(t + dt) * dt
            accel = s.forces / masses
            s.vel += 0.5 * dt * accel

        Ekin = np.array([v.item() for v in kinetic_energy(masses, s.vel)])
        T_curr = kinetic_to_temp(Ekin, self.systems.dof)
        return Ekin, pot.cpu().numpy(), T_curr

class MTSIntegrator(Integrator):
    """
    Multiple Time Step (MTS) Integrator.
    Uses an Impulse-RESPA approach.
    Can also revert to NVE-MTS if T is None. This is based on https://arxiv.org/abs/2412.11569 to use slow force corrections.
    """
    def __init__(self, systems, forces, slow_forces, timestep, device, gamma=None, T=None, integrate_force=False, mts_framestep=0):
        super().__init__(systems, forces, slow_forces, timestep, device, gamma, T, integrate_force, mts_framestep)
    
    def step(self, niter=1, curr_step=0):
        s = self.systems
        masses = self.forces.par.masses
        dt = self.dt
        M = self.mts_framestep
        
        for step in range(curr_step, curr_step + niter):
            is_mts_start = (step % M) == 0
            
            # --- Slow Force Impulse (Start) ---
            if is_mts_start and self._f_correction is None:
                _, fast_force_check, _ = self.slow_forces.compute(
                    s.pos, s.box, s.forces, toNumpy=False, calculateForces=False
                )
                # Correction = Total_Slow - Fast_Approximation
                self._f_correction = fast_force_check - s.forces 
                s.vel += 0.5 * (dt * M) * (self._f_correction / masses)

            # --- Fast Inner Loop (Velocity Verlet / BAOAB) ---
            # 1. Half Kick Fast
            s.vel += 0.5 * dt * (s.forces / masses)
            
            # 2. Drift Fast
            s.pos += s.vel * dt
            
            # 3. Force Update Fast
            pot, s.forces = self._compute_forces(s.pos, s.forces)

            # 4. Thermostat (Optional)
            if self.T is not None:
                self._ou_step(s.vel)

            # 5. Half Kick Fast
            s.vel += 0.5 * dt * (s.forces / masses)
            
            # --- Slow Force Impulse (End) ---
            if ((step + 1) % M) == 0:
                s.vel += 0.5 * (dt * M) * (self._f_correction / masses)
                self._f_correction = None 
        
        Ekin = np.array([v.item() for v in kinetic_energy(masses, s.vel)])
        T_curr = kinetic_to_temp(Ekin, self.systems.dof)
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
            pot, s.forces = self._compute_forces(s.pos, s.forces)
            
            # B: Second half kick
            s.vel += 0.5 * dt * (s.forces / masses)
        
        Ekin = np.array([v.item() for v in kinetic_energy(masses, s.vel)])
        T_curr = kinetic_to_temp(Ekin, self.systems.dof)
        return Ekin, pot.cpu().numpy(), T_curr

INTEGRATOR_MAP = {
    "Langevin": LangevinIntegrator,
    "LangevinMiddle": LangevinMiddleIntegrator,
    "MTS": MTSIntegrator,
    "BAOAB": BAOABIntegrator,
}

def get_integrator(name, systems, forces, timestep, device, gamma=None, T=None, **kwargs):
    """
    Factory function to initialize an integrator by name.
    
    Args:
        name (str): One of 'Langevin', 'LangevinMiddle', 'MTS', 'BAOAB'
        **kwargs: Additional arguments passed to the integrator (e.g. slow_forces, mts_framestep)
    """
    if name not in INTEGRATOR_MAP:
        raise ValueError(f"Unknown integrator: {name}. Available: {list(INTEGRATOR_MAP.keys())}")
    
    integrator_class = INTEGRATOR_MAP[name]
    import inspect
    sig = inspect.signature(integrator_class.__init__)
    param_names = sig.parameters.keys()
    if 'slow_forces' in param_names and 'slow_forces' not in kwargs:
        raise ValueError(f"Integrator {name} requires 'slow_forces' argument.")
    else:
        kwargs.pop('slow_forces', None)
        kwargs.pop('mts_framestep', None)
        
    return integrator_class(
        systems=systems, 
        forces=forces, 
        timestep=timestep, 
        device=device, 
        gamma=gamma, 
        T=T, 
        **kwargs
    )
