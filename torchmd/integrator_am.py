import numpy as np
import torch

TIMEFACTOR = 48.88821
BOLTZMAN = 0.001987191


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
    """
    Simple MD integrator supporting velocity Verlet integration and two flavors of
    Langevin thermostats.  If a friction coefficient (`gamma`) and bath
    temperature (`T`) are provided, the integrator applies Langevin dynamics.  If
    `langevin_middle` is True, the stochastic dynamics are integrated using
    the BAOAB/Langevin-middle (LF middle) scheme, which exactly integrates
    the Ornstein-Uhlenbeck part via an Ornstein-Uhlenbeck update at the
    beginning and end of the step.  Otherwise, a simple Euler-Maruyama
    scheme is used (matching the original Langevin implementation).

    In addition, optional removal of rigid-body modes can be applied after
    force evaluation: zeroing the net force on the system (translation) and
    removing net torque about the centre of mass.  This can improve the
    stability of simulations with machine-learned force fields, where the
    predicted forces may contain small global bias terms that otherwise lead
    to unphysical drift during a trajectory.
    """

    def __init__(
        self,
        systems,
        forces,
        timestep,
        device,
        gamma=None,
        T=None,
        batch=None,
        integrate_force=False,
        *,
        langevin_middle: bool = False,
        remove_com: bool = False,
        remove_torque: bool = False,
    ):
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
            self.masses = self.forces.par.masses
            self.masses = torch.tensor(
                self.masses, device=device, dtype=systems.pos.dtype
            )
            self.masses = self.masses.view(-1, 1)

        if T and gamma is not None:
            # Precompute coefficient for Euler–Maruyama noise (old scheme).  In
            # the middle-Langevin scheme the noise coefficients are computed
            # on-the-fly each step based on exp(-gamma*dt/2).
            self.vcoeff = torch.sqrt(
                2.0 * gamma / self.masses * BOLTZMAN * T * self.dt
            ).to(device)
        self.batch = batch
        if batch is not None:
            # number of atoms per batch
            self.natoms = torch.bincount(batch).cpu().numpy()
        else:
            self.natoms = len(self.masses)
        
        # store curl during integration when compute_curl is true
        self.curl_storage = []

        # whether to use the middle discretization for Langevin dynamics
        self.langevin_middle = langevin_middle and gamma is not None and T is not None
        # whether to remove net translation and/or net torque from the forces
        self.remove_com = remove_com
        self.remove_torque = remove_torque

    def _remove_rigid_body(self, forces: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Optionally remove rigid body modes from the forces.  If ``remove_com`` is
        set, the mean force over all atoms is subtracted from each atom to
        enforce zero net force.  If ``remove_torque`` is set, the forces are
        adjusted to cancel the net torque about the centre of mass.  The
        correction for torque is computed by solving a 3x3 linear system per
        replica: given relative positions ``r_rel`` and current net torque
        ``tau``, we solve ``A*K = -tau``, where ``A`` is the symmetric matrix
        \(\sum_i (|r_i|^2 I - r_i r_i^T)\).  The correction on force i is then
        \(\Delta f_i = K \times r_i\).  This algorithm preserves net force
        automatically and minimally perturbs the forces in a least-squares sense.

        Parameters
        ----------
        forces : torch.Tensor
            Current forces, shape (nreplicas, natoms, 3).
        pos : torch.Tensor
            Current positions, shape (nreplicas, natoms, 3).

        Returns
        -------
        torch.Tensor
            Corrected forces with rigid body modes removed.
        """
        # Nothing to do if neither translation nor rotation removal is requested
        if not (self.remove_com or self.remove_torque):
            return forces

        # Ensure we operate in float64 for numerical stability during torque removal
        orig_dtype = forces.dtype
        f = forces.to(dtype=torch.float64)
        r = pos.to(dtype=torch.float64)
        nreplicas, natoms, _ = f.shape

        # Remove net force (translation) per replica
        if self.remove_com:
            f_mean = f.mean(dim=1, keepdim=True)
            f = f - f_mean

        # Remove net torque per replica
        if self.remove_torque:
            # Compute masses and centre of mass
            m = self.masses.to(dtype=torch.float64, device=f.device)  # (natoms,1)
            total_mass = torch.sum(m)
            m_r = m.view(1, natoms, 1)
            com = (r * m_r).sum(dim=1, keepdim=True) / total_mass
            r_rel = r - com
            # Build inertia-like matrix A for each replica
            # A[k] = sum_i (|r_i|^2 * I - r_i ⊗ r_i)
            A = torch.zeros((nreplicas, 3, 3), dtype=torch.float64, device=f.device)
            # vectorized accumulation
            # r_rel_sq: (nreplicas, natoms, 1)
            r_rel_sq = torch.sum(r_rel * r_rel, dim=2, keepdim=True)
            # outer products: (nreplicas, natoms, 3, 3)
            outer = r_rel.unsqueeze(3) * r_rel.unsqueeze(2)
            # identity matrix broadcast: (1,1,3,3)
            eye = torch.eye(3, dtype=torch.float64, device=f.device).view(1, 1, 3, 3)
            A = torch.sum(r_rel_sq.unsqueeze(-1) * eye - outer, dim=1)
            # current net torque per replica: tau[k] = sum_i r_rel_i × f_i
            tau = torch.sum(torch.cross(r_rel, f, dim=2), dim=1)
            # Solve A K = -tau for each replica using pseudo-inverse to handle singular cases
            # K shape (nreplicas, 3)
            K = torch.zeros_like(tau, dtype=torch.float64)
            for k in range(nreplicas):
                # Use pinv for robustness; add tiny damping if necessary
                A_inv = torch.linalg.pinv(A[k])
                K[k] = A_inv @ (-tau[k])
            # Compute correction to forces: delta_f_i = K × r_rel_i
            # Expand K to (nreplicas, natoms, 3)
            K_exp = K.unsqueeze(1).expand(-1, natoms, -1)
            delta_f = torch.cross(K_exp, r_rel, dim=2)
            f = f + delta_f
        # Restore original dtype
        return f.to(orig_dtype)

    def step(self, niter: int = 1):
        """
        Advance the system by ``niter`` steps.

        If both ``gamma`` and ``T`` are specified, Langevin dynamics are used.  If
        ``langevin_middle`` is True, the integration uses the BAOAB/Langevin
        middle discretization; otherwise, an Euler-Maruyama update is applied
        after the velocity Verlet force update.  For purely Hamiltonian
        dynamics (``gamma`` or ``T`` is None), a standard velocity Verlet
        integrator is used.

        At the end of each step, the kinetic energy and instantaneous
        temperature are computed and returned along with the potential energy.

        Returns
        -------
        tuple
            (Ekin, pot, T) where Ekin and T are numpy arrays of kinetic
            energies and temperatures per replica (possibly per batch if
            batching is used).
        """
        systems = self.systems

        pot = 0.0  # initialise potential outside loop
        # Precompute constants for middle integrator if needed
        if self.langevin_middle:
            # friction coefficient is per unit time; dt in our units.  Compute
            # half-step factors for Ornstein–Uhlenbeck update.  Use dtype of
            # velocities.
            gamma_dt_half = self.gamma * self.dt / 2.0
            a = torch.exp(-gamma_dt_half)
            # noise coefficient b depends on masses and temperature.  Shape
            # (natoms,1).  We broadcast across replicas later.
            b = torch.sqrt(
                BOLTZMAN * self.T * (1.0 - a * a) / self.masses
            ).to(self.device)

        for _ in range(niter):
            if self.gamma is not None and self.T is not None:
                if self.langevin_middle:
                    # Langevin middle (BAOAB) scheme
                    # Generate random noise for the pre-force OU step
                    noise = torch.randn_like(systems.vel)
                    # Expand a and b to (nreplicas, natoms, 1)
                    a_view = a.view(1, 1, 1)
                    b_view = b.view(1, -1, 1)
                    systems.vel = a_view * systems.vel + b_view * noise
                    # Kick half-step with current forces
                    systems.vel += 0.5 * self.dt * systems.forces / self.masses
                    # Drift: update positions
                    systems.pos += self.dt * systems.vel
                    # Compute new forces and (optionally) curl
                    pot, curl = self.forces.compute(systems.pos, systems.box, systems.forces)
                    if curl is not None:
                        self.curl_storage.append(curl)
                    # Remove rigid-body modes from forces if requested
                    if self.remove_com or self.remove_torque:
                        systems.forces = self._remove_rigid_body(systems.forces, systems.pos)
                    # Kick second half-step with updated forces
                    systems.vel += 0.5 * self.dt * systems.forces / self.masses
                    # Final OU step
                    noise = torch.randn_like(systems.vel)
                    systems.vel = a_view * systems.vel + b_view * noise
                else:
                    # Euler–Maruyama Langevin update (previous implementation)
                    _first_VV(systems.pos, systems.vel, systems.forces, self.masses, self.dt)
                    pot, curl = self.forces.compute(systems.pos, systems.box, systems.forces)
                    if curl is not None:
                        self.curl_storage.append(curl)
                    # Remove rigid-body modes if requested
                    if self.remove_com or self.remove_torque:
                        systems.forces = self._remove_rigid_body(systems.forces, systems.pos)
                    # Apply friction and noise
                    langevin(systems.vel, self.gamma, self.vcoeff, self.dt, self.device)
                    _second_VV(systems.vel, systems.forces, self.masses, self.dt)
            else:
                # Hamiltonian (velocity Verlet) integration
                _first_VV(systems.pos, systems.vel, systems.forces, self.masses, self.dt)
                pot, curl = self.forces.compute(systems.pos, systems.box, systems.forces)
                if curl is not None:
                    self.curl_storage.append(curl)
                # Remove rigid-body modes
                if self.remove_com or self.remove_torque:
                    systems.forces = self._remove_rigid_body(systems.forces, systems.pos)
                _second_VV(systems.vel, systems.forces, self.masses, self.dt)

        # Compute kinetic energies and temperatures for all replicas (and batches)
        ke_result = kinetic_energy(self.masses, systems.vel, self.batch)
        Ekin = ke_result.flatten().cpu().numpy()
        Tinst = kinetic_to_temp(Ekin, self.natoms)
        return Ekin, pot, Tinst
