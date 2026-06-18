import numpy as np
import torch

from torchmdnet.calculators import CompileExternal as _BaseCompileExternal


class CompileExternal:
    def __init__(
        self,
        netfile,
        embeddings,
        device="cpu",
        to_constraint_harmonic=None,
        k_harmonic=1.0,
        to_constraint_flat_bottom_box=None,
        k_flat_bottom=0.1,
        pocket_idxs=None,
        c7_index=None,
        **kwargs,
    ):
        self.base = _BaseCompileExternal(
            netfile,
            embeddings,
            device=device,
            **kwargs,
        )
        self.model = self.base.model
        self.device = self.base.device
        self.dtype = self.base.dtype
        self.n_atoms = self.base.n_atoms

        self.harmonic_indices = None
        self.harmonic_k = None
        self.reference_positions = None

        self.flat_bottom_half_widths = None
        self.flat_bottom_k = None
        self.pocket_indices = None
        self.c7_index = None

        if to_constraint_harmonic is not None:
            harmonic_indices = np.load(to_constraint_harmonic).astype(np.int64)
            self.harmonic_indices = torch.tensor(
                harmonic_indices, dtype=torch.long, device=self.device
            )
            self.harmonic_k = torch.tensor(
                k_harmonic,
                dtype=self.dtype,
                device=self.device,
            )

        if to_constraint_flat_bottom_box is not None or pocket_idxs is not None or c7_index is not None:
            if to_constraint_flat_bottom_box is None or pocket_idxs is None or c7_index is None:
                raise ValueError(
                    "Flat-bottom restraints require `to_constraint_flat_bottom_box`, `pocket_idxs`, and `c7_index`."
                )
            self.flat_bottom_half_widths = torch.tensor(
                0.5 * np.asarray(to_constraint_flat_bottom_box, dtype=np.float64),
                dtype=self.dtype,
                device=self.device,
            )
            self.flat_bottom_k = torch.tensor(
                k_flat_bottom,
                dtype=self.dtype,
                device=self.device,
            )
            self.pocket_indices = torch.tensor(
                np.load(pocket_idxs).astype(np.int64),
                dtype=torch.long,
                device=self.device,
            )
            self.c7_index = int(c7_index)

    def _ensure_reference_positions(self, pos):
        if self.harmonic_indices is not None and self.reference_positions is None:
            self.reference_positions = pos.detach().clone()

    def _apply_harmonic_restraint(self, pos, energy, forces):
        if self.harmonic_indices is None:
            return energy, forces

        ref = self.reference_positions.index_select(1, self.harmonic_indices)
        current = pos.index_select(1, self.harmonic_indices)
        delta = current - ref
        energy = energy + 0.5 * self.harmonic_k * torch.sum(delta * delta, dim=(1, 2))
        forces.index_add_(1, self.harmonic_indices, -self.harmonic_k * delta)
        return energy, forces

    def _apply_flat_bottom_restraint(self, pos, energy, forces):
        if self.flat_bottom_half_widths is None:
            return energy, forces

        c7 = pos[:, self.c7_index, :]
        pocket = pos.index_select(1, self.pocket_indices)
        centroid = pocket.mean(dim=1)
        delta = c7 - centroid

        abs_delta = torch.abs(delta)
        excess = torch.clamp(abs_delta - self.flat_bottom_half_widths, min=0.0)
        energy = energy + 0.5 * self.flat_bottom_k * torch.sum(excess * excess, dim=1)

        sign = torch.sign(delta)
        force_on_c7 = -self.flat_bottom_k * excess * sign
        forces[:, self.c7_index, :] += force_on_c7

        pocket_share = -force_on_c7 / float(self.pocket_indices.numel())
        pocket_share = pocket_share.unsqueeze(1).expand(-1, self.pocket_indices.numel(), -1)
        scatter_index = self.pocket_indices.view(1, -1, 1).expand(pos.shape[0], -1, 3)
        forces.scatter_add_(1, scatter_index, pocket_share)
        return energy, forces

    def calculate(self, pos, box=None):
        self._ensure_reference_positions(pos)

        energy, forces = self.base.calculate(pos, box=box)
        energy = energy.to(self.device).to(self.dtype).reshape(-1)
        forces = forces.to(self.device).to(self.dtype)

        energy, forces = self._apply_harmonic_restraint(pos, energy, forces)
        energy, forces = self._apply_flat_bottom_restraint(pos, energy, forces)
        return energy, forces
