from scipy import constants as const
import torch
import numpy as np
from math import pi


class NNPForces:
    """
    Parameters
    ----------

    """
    def __init__(
        self,
        parameters,
        external=None,
        return_forces=True,   # If True, return forces instead of potential energy
    ):
        assert external is not None, "An external potential must be provided for NNPForces. Otherwise use Forces class."
        self.return_forces = return_forces
        self.par = parameters

        self.natoms = len(parameters.masses)
        self.external = external

    def compute(
        self,
        pos,
        box,
        forces,
        returnDetails=False,
        explicit_forces=True,
        toNumpy=True,
        calculateForces=True,
    ):

        if self.return_forces:
            if not calculateForces and not explicit_forces:
                raise RuntimeError(
                    """To return forces explicit_forces must be True when calculateForces is False. Please set explicit_forces=True and 
                    be sure that the potentials return forces directly."""
                )
        elif calculateForces:
            if not explicit_forces and not pos.requires_grad:
                raise RuntimeError(
                    "The positions passed don't require gradients. Please use pos.detach().requires_grad_(True) before passing."
                )
        else:
            explicit_forces = False  
            
        nsystems = pos.shape[0]
        pot = torch.zeros(nsystems, device=pos.device).type(pos.dtype)
        all_curl = torch.zeros((nsystems, self.natoms, 3), device=pos.device).type(pos.dtype)
                    
        if forces is not None:
            forces.zero_()
            
        # force correction from: 'The dark side of the forces' https://arxiv.org/abs/2412.11569 (sec H)
        # conservative forces computed as -grad V (neg_dy) (Tensornet)
        # non-conservative forces computed directly from the model as equivariant-vector (v-Tensornet)
        # vec shape num_systems x num_atoms x 3
        # y: shape num_systems, 1
        y, vec = self.external.calculate(pos, box=None)

        # if is_conservative, then y: energy, vec: -dy
        # if not is_conservative, then y: curl (if compute_curl true), vec: vector output forces
        is_conservative = not self.external.model.non_conservative
        if is_conservative:
            if len(vec) > 0:
                # if model derivative is false, vec is torch.empty so update is skipped
                forces[:] = vec
                
            for s in range(nsystems):
                pot[s] = y[s]
        else:
            # if the model is non-conservative, vec are the forces directly and always computed
            forces[:] = vec
            if self.external.model.inference_curl:
                for s in range(nsystems):
                    all_curl[s] = y[s]
                                                
        if toNumpy:
            return pot.detach().cpu().numpy(), forces.detach().cpu().numpy(), all_curl.detach().cpu().numpy()
        return pot, forces, all_curl