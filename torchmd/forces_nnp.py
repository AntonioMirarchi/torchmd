import torch


class NNPForces:
    """MLIP-only force provider aligned with the ``Forces.compute`` contract.

    The external calculator owns differentiation.  In particular,
    ``CompileExternal`` evaluates a scalar energy and returns ``-dE/dx``.  The
    returned energy and force tensors are intentionally detached so that an MD
    step does not retain the model's autograd graph.

    Behavior
    --------
    * calculateForces=False:
        - Do NOT modify the `forces` buffer (no zeroing, no overwrite).
        - Still evaluates the MLIP once to obtain energies.

    * calculateForces=True:
        - Forces are copied from the external calculator's force output.
        - If the external calculator does not return forces, this raises.

    Returns
    -------
    Matches Forces.compute:
      - pot: np.ndarray of shape (nsystems,)
      - forces: modified in-place buffer
    """

    def __init__(
        self,
        parameters,
        external=None,
        calculateForces=True,
        explicit_forces=True,
    ):
        assert external is not None, (
            "An external MLIP must be provided for NNPForces. "
            "Otherwise use the standard Forces class."
        )
        self.par = parameters
        self.natoms = len(parameters.masses)
        self.external = external
        self.calculateForces = calculateForces
        if calculateForces and not explicit_forces:
            raise ValueError(
                "NNPForces expects the external calculator to return forces. "
                "For CompileExternal, use explicit_forces=True: the calculator "
                "computes -dE/dx internally."
            )
        self.explicit_forces = explicit_forces

    def compute(
        self,
        pos,
        box,
        forces,
        toNumpy=True,
    ):
        nsystems = pos.shape[0]

        # Periodic boxes are currently disabled by run.py. Keep the explicit
        # argument here so this adapter remains correct when they are enabled.
        ext_ene, ext_force = self.external.calculate(pos, box=box)
        pot = ext_ene.reshape(-1).to(device=pos.device, dtype=pos.dtype)
        if pot.numel() != nsystems:
            raise RuntimeError(
                f"MLIP returned {pot.numel()} energies for {nsystems} systems."
            )

        if self.calculateForces:
            if ext_force is None or ext_force.numel() == 0:
                raise RuntimeError(
                    "The external MLIP calculator did not return forces."
                )
            if ext_force.shape != forces.shape:
                raise RuntimeError(
                    f"MLIP forces shape {tuple(ext_force.shape)} does not match "
                    f"forces buffer shape {tuple(forces.shape)}."
                )

            # The integrator owns this persistent buffer. Do not attach the
            # calculator's autograd graph to it.
            forces.copy_(
                ext_force.detach().to(device=forces.device, dtype=forces.dtype)
            )

        # Return energies (and optional details) only.
        if toNumpy:
            return pot.detach().cpu().numpy()

        return pot
