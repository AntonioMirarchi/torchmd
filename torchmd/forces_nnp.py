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
        integrate_neg_dy=False,
    ):
        assert external is not None, "An external potential must be provided for NNPForces. Otherwise use Forces class."
        self.return_forces = return_forces
        self.integrate_neg_dy = integrate_neg_dy # If True, integrate -dy instead of vec when the model is non-conservative
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
            
        # if self.external.model.compute_curl the curl is computed and returned
        # if self.compute_grad then y and -dy are returned otherwise only empty tensors
        # vec always contains the forces and always returned
        y, neg_dy, vec, curl = self.external.calculate(pos, box=None)

        is_conservative = not self.external.model.non_conservative
        if is_conservative:
            if len(neg_dy) > 0:
                # if model derivative is false, vec is torch.empty so update is skipped
                forces[:] = neg_dy
                
            for s in range(nsystems):
                pot[s] = y[s]
            
        else:
            # if the model is non-conservative, vec are the forces directly and always computed
            if self.integrate_neg_dy:
                assert len(neg_dy) > 0, "neg_dy is empty but integrate_neg_dy is True."
                forces[:] = neg_dy
            else:
                forces[:] = vec
            if self.external.model.compute_curl:
                for s in range(nsystems):
                    all_curl[s] = curl[s]
            if self.external.model.compute_grad:
                for s in range(nsystems):
                    pot[s] = y[s]
                                                
        if toNumpy:
            return pot.detach().cpu().numpy(), forces.detach().cpu().numpy(), all_curl.detach().cpu().numpy()
        return pot, forces, all_curl