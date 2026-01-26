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

    def __init__(self, parameters, external=None):
        assert external is not None, (
            "An external MLIP must be provided for NNPForces. "
            "Otherwise use the standard Forces class."
        )
        self.par = parameters
        self.natoms = len(parameters.masses)
        self.external = external

    def compute(
        self,
        pos,
        box,
        forces,
        toNumpy=True,
        calculateForces=True,
        # kept ONLY for API compatibility with callers using Forces.compute signature
        # (ignored in MLIP-only pathway)
        explicit_forces=None,
    ):
        nsystems = pos.shape[0]

        pot = torch.zeros(nsystems, device=pos.device, dtype=pos.dtype)
        all_curl = torch.zeros((nsystems, self.natoms, 3), device=pos.device, dtype=pos.dtype)
        curl_computed = False
        (y, pred_forces), curl = self.external.calculate(pos, box=None)
        is_conservative = not self.external.model.non_conservative

        if is_conservative:
            # y is energy per system (shape: [nsystems] or [nsystems, 1])
            pot[:] = y.reshape(-1)

        if curl.numel() > 0:
                curl = curl.unsqueeze(0) 
                # expect y shape [nsystems, natoms, 3]
                all_curl[:] = curl
                curl_computed = True

        # Only touch the force output buffer if requested
        if calculateForces and (forces is not None):
            # MLIP forces must come from vec in the MLIP-only setup.
            if pred_forces is None or (hasattr(pred_forces, "numel") and pred_forces.numel() == 0):
                raise RuntimeError(
                    "MLIP did not return forces (empty `vec`) but calculateForces=True. "
                    "Enable force output in the MLIP or set calculateForces=False."
                )
            forces.zero_()
            forces[:] = pred_forces

        # Return energies (and optional details) only.
        if toNumpy:
            pot_np = pot.detach().cpu().numpy()
            if curl_computed:
                return pot_np, all_curl.detach().cpu().numpy()
            return pot_np, None

        if curl_computed:
            return pot, all_curl.detach().cpu().numpy()
        return pot, None
