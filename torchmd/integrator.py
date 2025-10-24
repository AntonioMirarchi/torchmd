import numpy as np
import torch

TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191


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


def kinetic_to_temp(Ekin, natoms):
    return 2.0 / (3.0 * natoms * BOLTZMAN) * Ekin


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


class Integrator:
    def __init__(self, systems, forces, timestep, device, gamma=None, T=None, integrate_force=False):
        """ Integrator for molecular dynamics simulations.
        Parameters:
        -----------
        systems : Systems
            The systems to be simulated.
        forces : Forces
            The forces acting on the systems.
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
        -----------
        """
        self.dt = timestep / TIMEFACTOR
        self.systems = systems
        self.forces = forces
        self.device = device
        gamma = gamma / PICOSEC2TIMEU
        self.gamma = gamma
        self.T = T
        
        if integrate_force:
            assert self.forces.return_forces, "For integrate_force=True, forces must return direct forces. Set return_forces=True in Forces."
            
        self.integrate_force = integrate_force
        if T:
            M = self.forces.par.masses
            self.vcoeff = torch.sqrt(2.0 * gamma / M * BOLTZMAN * T * self.dt).to(
                device
            )

    def step(self, niter=1):
        s = self.systems
        masses = self.forces.par.masses
        natoms = len(masses)
        for _ in range(niter):
            _first_VV(s.pos, s.vel, s.forces, masses, self.dt) # first half update
            
            if self.integrate_force:
                pot, upd_forces = self.forces.compute(s.pos, s.box, s.forces) # directly get forces. pot is None
                s.forces = upd_forces # this will be used in the second half update
                pot = None
            else:
                pot, _ = self.forces.compute(s.pos, s.box, s.forces) # update forces internally for the second half update. forces output is None
            
            if self.T:
                langevin(s.vel, self.gamma, self.vcoeff, self.dt, self.device)
            
            _second_VV(s.vel, s.forces, masses, self.dt)

        Ekin = np.array([v.item() for v in kinetic_energy(masses, s.vel)])
        T = kinetic_to_temp(Ekin, natoms)
        return Ekin, pot, T
