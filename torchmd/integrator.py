import numpy as np
import torch

# Internal units are Angstrom, kcal/mol, atomic mass units, and TIMEFACTOR fs.
# With these units F / m is acceleration in Angstrom / internal_time**2.
TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191  # kcal mol^-1 K^-1


def kinetic_energy(masses, vel, batch=None):
    """
    Kinetic energy calculation for molecular dynamics format.

    Args:
        masses: Mass tensor of shape (natoms, 1)
        vel: Velocity tensor of shape (nreplicas, natoms, 3)
        batch: Optional batch indices of shape (natoms,) grouping atoms within each replica into batches

    Returns:
        If batch is None: kinetic energy per replica, shape (nreplicas, 1)
        If batch is provided: kinetic energy per replica per batch, shape (nreplicas, nbatches)
    """
    if vel.dim() != 3:
        raise ValueError(f"vel must be 3D (nreplicas, natoms, 3), got {vel.dim()}D")

    # Calculate per-atom kinetic energies: (nreplicas, natoms, 1)
    v_sq = torch.sum(vel * vel, dim=2, keepdim=True)  # (nreplicas, natoms, 1)
    E_per_atom = 0.5 * masses * v_sq  # (nreplicas, natoms, 1)

    if batch is None:
        # Sum over atoms for each replica: (nreplicas, 1)
        return torch.sum(E_per_atom, dim=1)

    # Batch atoms within each replica
    n_batch = int(torch.max(batch).item() + 1)
    nreplicas, natoms = vel.shape[0], vel.shape[1]

    # Initialize result: (nreplicas, nbatches)
    Ekin = torch.zeros(nreplicas, n_batch, device=vel.device, dtype=vel.dtype)

    # For each replica, accumulate kinetic energies by batch
    for r in range(nreplicas):
        Ekin[r].index_add_(0, batch, E_per_atom[r, :, 0])

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


def _kick(vel, force, mass, dt):
    """Advance velocities by ``dt`` using the current force (B operator)."""
    vel.add_(force / mass, alpha=dt)


def _drift(pos, vel, dt):
    """Advance positions by ``dt`` using the current velocity (A operator)."""
    pos.add_(vel, alpha=dt)


def langevin(vel, damping, noise_scale):
    """Apply the exact Ornstein--Uhlenbeck velocity update in place."""
    vel.mul_(damping).add_(torch.randn_like(vel) * noise_scale)


PICOSEC2TIMEU = 1000.0 / TIMEFACTOR


class Integrator:
    """Velocity-Verlet NVE and BAOAB (Langevin-middle) NVT integrator.

    ``systems.forces`` must contain the force at ``systems.pos`` on entry to
    :meth:`step`. After every iteration it contains the force at the updated
    positions. Velocities are stored at integer time, not at a staggered
    half-step.

    NVE applies ``B(dt/2) A(dt) B(dt/2)``. NVT applies the symmetric
    Langevin-middle/BAOAB splitting
    ``B(dt/2) A(dt/2) O(dt) A(dt/2) B(dt/2)``.
    """

    def __init__(
        self, systems, forces, timestep, device, gamma=None, T=None, batch=None,
    ):
        if timestep <= 0:
            raise ValueError(f"timestep must be positive, got {timestep}")
        if (gamma is None) != (T is None):
            raise ValueError("gamma and T must either both be set or both be None")
        if gamma is not None and gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if T is not None and T < 0:
            raise ValueError(f"T must be non-negative, got {T}")

        self.dt = timestep / TIMEFACTOR
        self.systems = systems
        self.forces = forces
        self.device = device
        if gamma is not None:
            gamma = gamma / PICOSEC2TIMEU
        self.gamma = gamma
        self.T = T
        if torch.any(systems.masses != 0):
            self.masses = systems.masses
        else:
            # Parameters already stores a tensor; normalize device, dtype, shape.
            self.masses = self.forces.par.masses
            self.masses = self.masses.detach().clone().to(
                device=device, dtype=systems.pos.dtype
            )
            self.masses = self.masses.view(-1, 1)

        if torch.any(self.masses <= 0):
            raise ValueError("All atom masses must be positive")

        if T is not None and gamma is not None:
            # Exact OU solution: v' = c*v + sqrt((1-c^2) kT/m) R.
            self.langevin_damping = float(np.exp(-gamma * self.dt))
            self.vcoeff = torch.sqrt(
                (1.0 - self.langevin_damping**2)
                * BOLTZMAN
                * T
                / self.masses
            ).to(device=device, dtype=systems.vel.dtype)
        self.batch = batch
        if batch is not None:
            # number of atoms per batch
            self.natoms = torch.bincount(batch).cpu().numpy()
        else:
            self.natoms = len(self.masses)

    def step(self, niter=1):
        if niter < 1:
            raise ValueError(f"niter must be at least 1, got {niter}")
        systems = self.systems

        for _ in range(niter):
            if self.gamma is not None and self.T is not None:
                # B(dt/2): first force half-kick.
                _kick(
                    systems.vel,
                    systems.forces,
                    self.masses,
                    0.5 * self.dt,
                )

                # A(dt/2) O(dt) A(dt/2): drift, exact OU thermostat, drift.
                _drift(systems.pos, systems.vel, 0.5 * self.dt)
                langevin(systems.vel, self.langevin_damping, self.vcoeff)
                _drift(systems.pos, systems.vel, 0.5 * self.dt)

                # Refresh F(x[t+dt]) in place, then B(dt/2).
                pot = self.forces.compute(
                    systems.pos, systems.box, systems.forces
                )
                _kick(
                    systems.vel,
                    systems.forces,
                    self.masses,
                    0.5 * self.dt,
                )
            else:
                # Canonical velocity Verlet: B(dt/2) A(dt) B(dt/2).
                _kick(
                    systems.vel,
                    systems.forces,
                    self.masses,
                    0.5 * self.dt,
                )
                _drift(systems.pos, systems.vel, self.dt)
                pot = self.forces.compute(
                    systems.pos, systems.box, systems.forces
                )
                _kick(
                    systems.vel,
                    systems.forces,
                    self.masses,
                    0.5 * self.dt,
                )

        ke_result = kinetic_energy(self.masses, systems.vel, self.batch)
        Ekin = ke_result.detach().flatten().cpu().numpy()
        if self.batch is None:
            T = 2.0 * Ekin / (self.systems.dof * BOLTZMAN)
        else:
            T = kinetic_to_temp(Ekin, self.natoms)
        return Ekin, pot, T
