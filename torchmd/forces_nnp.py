from scipy import constants as const
import torch
import numpy as np
from math import pi

class NNPForces:
    """MLIP-only force provider aligned with the `Forces.compute` contract.

    Behavior
    --------
    * calculateForces=False:
        - Do NOT modify the `forces` buffer (no zeroing, no overwrite).
        - Still evaluates the MLIP once to obtain energies/curl outputs.

    * calculateForces=True:
        - Forces are taken from the MLIP vector output `vec` (for both conservative and non-conservative).
        - If the MLIP does not return `vec` (empty), this raises.

    Returns
    -------
    Matches Forces.compute:
      - pot: np.ndarray of shape (nsystems,)
      - forces: modified in-place buffer
      - curl: np.ndarray of shape (nsystems, natoms, 3) if computed, else None
    """

    def __init__(self, 
                parameters, 
                external=None,
                calculateForces=True, # (need explicit_forces false) and the pos to requires_grad to True
                explicit_forces=None, # This means that the external MLIP is expected to return forces
):
        assert external is not None, (
            "An external MLIP must be provided for NNPForces. "
            "Otherwise use the standard Forces class."
        )
        self.par = parameters
        self.natoms = len(parameters.masses)
        self.external = external
        self.calculateForces = calculateForces
        self.explicit_forces = explicit_forces

    def compute(
        self,
        pos,
        box,
        forces,
        toNumpy=True,
    ):
        if self.calculateForces:
            if not self.explicit_forces and not pos.requires_grad:
                raise RuntimeError(
                    "The positions passed don't require gradients. Please use pos.detach().requires_grad_(True) before passing."
                )
        else:
            self.explicit_forces = False
        nsystems = pos.shape[0]

        pot = torch.zeros(nsystems, device=pos.device, dtype=pos.dtype)

        ext_ene, ext_force = self.external.calculate(pos, box=None) #box=box)
        is_conservative = not self.external.model.non_conservative

        if is_conservative:
            # ext_ene is energy per system (shape: [nsystems] or [nsystems, 1])
            pot[:] = ext_ene.reshape(-1)
            
            if not self.explicit_forces and self.calculateForces:
                # Compute forces via autograd if not explicitly provided by the MLIP.
                # This is only valid for conservative models.
                ext_force = -torch.autograd.grad(pot.sum(), pos, retain_graph=True)[0]

        
        # MLIP forces must come from vec in the MLIP-only setup.
        if ext_force is None or (hasattr(ext_force, "numel") and ext_force.numel() == 0):
            raise RuntimeError(
                "MLIP did not return forces (empty `vec`) but calculateForces=True. "
                "Enable force output in the MLIP or set calculateForces=False."
            )
        if ext_force.shape != forces.shape:
            raise RuntimeError(
                f"MLIP forces shape {tuple(ext_force.shape)} does not match "
                f"forces buffer shape {tuple(forces.shape)}."
            )
         
        # Update the forces buffer in-place with the MLIP forces.   
        forces.zero_()
        forces[:] = ext_force

        # Return energies (and optional details) only.
        if toNumpy:
            pot_np = pot.detach().cpu().numpy()
            return pot_np

        return pot
