"""Independent THP replicas: original losses, vmap execution, packed Adam."""
from copy import deepcopy

import torch
from torch import nn
from torch.func import functional_call, stack_module_state

from model.model import Transformer
from train.train import VARIANTS, batch_loss, model_config

STACKABLE_VARIANTS = ("A0", "A1", "A2", "A3", "A3J")


class Objective(nn.Module):
    """Keep forward, prediction and parameter-dependent THP loss in one call."""

    def __init__(self, model: Transformer, variant: str):
        super().__init__()
        self.model = model
        self.activity_loss = VARIANTS[variant][1]

    def forward(self, batch: dict, prediction: bool = False):
        if not prediction:
            act, tim, _ = batch_loss(self.model, batch, self.activity_loss, self.model.n_mix)
            return torch.stack((act, tim))
        h, heads = self.model(batch["types"], batch["event_time"])
        baseline = self.model.model_type == "baseline_act"
        if baseline:
            time = torch.cat((heads[0], self.model.linear(h),
                              batch["event_time"].unsqueeze(-1)), -1)
        else:
            time = torch.cat((heads[0].unsqueeze(-1), heads[1]), -1)
        return heads[-1].softmax(-1), time


def make_models(variant: str, size: str, seeds, num_types: int, dropout: float = .1, *, config=None):
    """Initialize on CPU without consuming the caller's random stream."""
    if variant not in STACKABLE_VARIANTS:
        raise ValueError(f"Standalone stacking supports {STACKABLE_VARIANTS}; use train.train for {variant}")
    expected = model_config(variant, size, num_types, dropout=dropout)
    config = expected if config is None else dict(config)
    if (config.keys() != expected.keys() or config["num_types"] != num_types or
            config["model_type"] != expected["model_type"]):
        raise ValueError("Exact model config must match the variant and activity vocabulary")
    return [Objective(Transformer.from_config(config, seed=seed), variant) for seed in seeds]


class PackedAdam:
    """Dense Adam for these models; each row has its own LR, moments and clock.

    Leaf parameter views share weights storage. Updating that storage avoids
    launching separate optimizer operations for every small parameter tensor.
    This implements the existing trainer's Adam (.9/.99, eps=1e-5), without
    weight decay, mixed precision or support for unused parameters.
    """

    def __init__(self, params: dict, lr: float = .002):
        n = next(iter(params.values())).shape[0]
        self.weights = torch.cat([p.detach().reshape(n, -1) for p in params.values()], 1)
        self.params, offset = {}, 0
        for name, value in params.items():
            count = value[0].numel()
            self.params[name] = nn.Parameter(self.weights[:, offset:offset + count].view_as(value))
            offset += count
        self.first = torch.zeros_like(self.weights)
        self.second = torch.zeros_like(self.weights)
        self.steps = self.weights.new_zeros(n)
        self.lr = self.weights.new_full((n,), lr)
        self.active = torch.ones(n, dtype=torch.bool, device=self.weights.device)

    def zero_grad(self):
        for parameter in self.params.values():
            parameter.grad = None

    @torch.no_grad()
    def step(self):
        if any(p.grad is None for p in self.params.values()):
            raise RuntimeError("PackedAdam requires a gradient for every parameter")
        gradient = torch.cat([p.grad.reshape(len(self.lr), -1) for p in self.params.values()], 1)
        norm = torch.linalg.vector_norm(gradient, dim=1)
        gradient.mul_((1 / (norm + 1e-6)).clamp(max=1)[:, None])
        active = self.active[:, None]
        self.steps.add_(self.active)
        self.first.copy_(torch.where(active, self.first.lerp(gradient, .1), self.first))
        self.second.copy_(torch.where(active, .99 * self.second + .01 * gradient.square(),
                                      self.second))
        clock = self.steps.clamp(min=1)
        denominator = self.second.sqrt() / (1 - .99 ** clock).sqrt()[:, None] + 1e-5
        update = (self.lr / (1 - .9 ** clock))[:, None] * self.first / denominator
        self.weights.sub_(torch.where(active, update, 0))
        return norm


class StackedModels:
    """Same-architecture models with an explicit leading replica dimension."""

    def __init__(self, models, randomness: str = "different"):
        if not models:
            raise ValueError("At least one model is required")
        params, self.buffers = stack_module_state(models)
        self.optimiser = PackedAdam(params)
        self.params = self.optimiser.params
        self.template = deepcopy(models[0]).to("meta")
        self.n = len(models)
        self._mapped = torch.vmap(self._call, in_dims=(0, 0, None, None),
                                  randomness=randomness)

    def _call(self, params, buffers, batch, prediction):
        return functional_call(self.template, (params, buffers), (batch, prediction))

    def train(self, enabled: bool):
        self.template.train(enabled)

    def losses(self, batch):
        return self._mapped(self.params, self.buffers, batch, False)

    def predictions(self, batch):
        return self._mapped(self.params, self.buffers, batch, True)

    def step(self, batch, targets: int, lam: float = 1.):
        self.optimiser.zero_grad()
        losses = self.losses(batch)
        individual = (losses[:, 0] + lam * losses[:, 1]) / max(targets, 1)
        # Sum across replicas: averaging would divide each model's gradient by n.
        (individual * self.optimiser.active).sum().backward()
        self.optimiser.step()
        return losses.detach()

    def configure(self, learning_rates, active):
        self.optimiser.lr.copy_(self.optimiser.lr.new_tensor(learning_rates))
        self.optimiser.active.copy_(self.optimiser.active.new_tensor(active))

    def state(self, replica: int | None = None):
        """Frozen CPU state, grouped unless one replica is requested."""
        return {name.removeprefix("model."): (value if replica is None else value[replica]).detach().cpu().clone()
                for name, value in (self.params | self.buffers).items()}

    @torch.no_grad()
    def restore(self, states):
        values = {name.removeprefix("model."): value for name, value in (self.params | self.buffers).items()}
        if not isinstance(states, dict):
            if len(states) != self.n or any(state.keys() != values.keys() for state in states):
                raise ValueError("Replica state keys or group size differ")
            states = {name: torch.stack([state[name] for state in states]) for name in values}
        if states.keys() != values.keys() or any(
                not isinstance(states[name], torch.Tensor) or states[name].shape != value.shape or
                states[name].dtype != value.dtype for name, value in values.items()):
            raise ValueError("Grouped state keys, shapes or dtypes differ")
        for name, value in values.items():
            value.copy_(states[name])

    def checkpoint(self):
        """Grouped model and packed Adam snapshot; model weights occur only once."""
        return {"model": self.state(), "parameter_names": list(self.params),
                "optimiser": {name: getattr(self.optimiser, name).detach().cpu().clone()
                              for name in ("first", "second", "steps", "lr", "active")}}

    @torch.no_grad()
    def load_checkpoint(self, checkpoint):
        if (checkpoint.keys() != {"model", "parameter_names", "optimiser"} or
                checkpoint["parameter_names"] != list(self.params)):
            raise ValueError("Checkpoint schema or parameter order differs")
        saved = checkpoint["optimiser"]
        names = ("first", "second", "steps", "lr", "active")
        if saved.keys() != set(names) or any(
                not isinstance(saved[name], torch.Tensor) or
                saved[name].shape != getattr(self.optimiser, name).shape or
                saved[name].dtype != getattr(self.optimiser, name).dtype for name in names):
            raise ValueError("Optimizer state keys, shapes or dtypes differ")
        self.restore(checkpoint["model"])
        for name in names:
            getattr(self.optimiser, name).copy_(saved[name])


