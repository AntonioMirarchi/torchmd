import numpy as np
import pytest
import torch

from torchmd.forces_nnp import NNPForces


class _Parameters:
    masses = torch.tensor([[12.0], [1.0]])


class _External:
    def __init__(self):
        self.seen_box = None

    def calculate(self, pos, box=None):
        self.seen_box = box
        # Emulate CompileExternal: force is derived internally and both outputs
        # returned to NNPForces are detached.
        local_pos = pos.detach().clone().requires_grad_(True)
        energy = (local_pos.square().sum(dim=(1, 2)))
        force = -torch.autograd.grad(energy.sum(), local_pos)[0]
        return energy.detach(), force.detach()


def test_nnp_forces_accepts_detached_compiled_calculator_outputs():
    external = _External()
    provider = NNPForces(_Parameters(), external=external)
    pos = torch.tensor(
        [[[1.0, 0.0, -2.0], [0.5, 1.5, 0.0]]], dtype=torch.float32
    )
    box = torch.eye(3).unsqueeze(0)
    force_buffer = torch.full_like(pos, float("nan"))

    energy = provider.compute(pos, box, force_buffer, toNumpy=False)

    torch.testing.assert_close(energy, pos.square().sum(dim=(1, 2)))
    torch.testing.assert_close(force_buffer, -2.0 * pos)
    assert external.seen_box is box
    assert not force_buffer.requires_grad


def test_nnp_forces_does_not_modify_buffer_when_disabled():
    provider = NNPForces(
        _Parameters(), external=_External(), calculateForces=False
    )
    pos = torch.ones(1, 2, 3)
    force_buffer = torch.full_like(pos, 7.0)

    energy = provider.compute(pos, None, force_buffer)

    np.testing.assert_allclose(energy, np.array([6.0], dtype=np.float32))
    torch.testing.assert_close(force_buffer, torch.full_like(pos, 7.0))


def test_nnp_forces_rejects_outer_autograd_mode():
    with pytest.raises(ValueError, match="external calculator to return forces"):
        NNPForces(
            _Parameters(),
            external=_External(),
            calculateForces=True,
            explicit_forces=False,
        )
