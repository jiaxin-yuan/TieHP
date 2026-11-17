import math

import torch
import torch.nn as nn

from model.Constants import PAD
from model.Layers import EncoderLayer

MODEL_TYPES = ("baseline_act", "mixture", "mixture_cause", "mixture_joint")


def get_non_pad_mask(seq):
    """[B, S] -> [B, S, 1], 1.0 on real events and 0.0 on PAD."""
    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """Mask the PAD keys for every query position."""
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD)
    return padding_mask.unsqueeze(1).expand(-1, len_q, -1)


def get_subsequent_mask(seq):
    """Upper-triangular mask hiding the future from each position."""
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    return subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)


class Encoder(nn.Module):
    """THP masked self-attention encoder; event times enter through a sinusoidal temporal encoding (no position index)."""

    def __init__(self, num_types, d_model, d_inner, n_layers, n_head, d_k, d_v, dropout):
        super().__init__()
        self.d_model = d_model
        self.register_buffer(   # temporal encoding basis
            'position_vec',
            torch.tensor([math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)]),
        )
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=PAD)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """Sinusoidal encoding of the event times: [B, S] -> [B, S, d_model]."""
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_type, event_time, non_pad_mask):
        """Encode an event sequence, masking the future and the padding."""
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        enc_output = self.event_emb(event_type)
        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
        return enc_output


class Predictor(nn.Module):
    """Original THP prediction head: linear without bias, xavier init, zero on PAD."""

    def __init__(self, dim, num_out):
        super().__init__()
        self.linear = nn.Linear(dim, num_out, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        """[B, S, dim] -> [B, S, num_out], zero on PAD."""
        return self.linear(data) * non_pad_mask


class Transformer(nn.Module):
    """THP encoder and heads, with explicit constructor/checkpoint dictionaries.

    Initialization preserves the original order and laws: embedding defaults;
    attention projection weights Xavier uniform; FFN/mixture heads Linear
    defaults; baseline time/activity heads bias-free Xavier normal; intensity
    Linear defaults with alpha=-.1 and beta=1. There is no second global reset.
    Sources: baseline_act heads = official THP (Predictor); mixture tie and
    time heads = THP-M transformer/model.py (nn.Linear defaults, bias). The
    activity head is THP's Predictor in every model type (2026-10-26), so the
    variants differ only in the components under test (not THP-M's nn.Linear).
    mixture_joint copies A3's time heads across activities at initialization,
    preserving shared seeded weights and starting at the independence submodel.
    """

    def __init__(self, num_types, d_model, d_inner, n_layers, n_head, d_k, d_v,
                 dropout=0.1, n_mix=5, model_type="mixture"):
        super().__init__()
        assert model_type in MODEL_TYPES, f"model_type must be one of {MODEL_TYPES}"
        self._config = dict(num_types=num_types, d_model=d_model, d_inner=d_inner,
                            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                            dropout=dropout, n_mix=n_mix, model_type=model_type)
        self.encoder = Encoder(num_types, d_model, d_inner, n_layers, n_head, d_k, d_v, dropout)
        self.num_types, self.n_mix, self.model_type = num_types, n_mix, model_type

        if model_type == "baseline_act":
            # original THP: intensity parameters, Δt head, activity head
            self.linear = nn.Linear(d_model, num_types)
            self.alpha = nn.Parameter(torch.tensor(-0.1))
            self.beta = nn.Parameter(torch.tensor(1.0))
            self.time_predictor = Predictor(d_model, 1)
            self.mark_head = Predictor(d_model, num_types)
        else:
            # zero atom: 1 tie logit (A1-A3) or logits over {zero by date, zero by batch, positive} (A4)
            self.tie_head = nn.Linear(d_model, 3 if model_type == "mixture_cause" else 1)
            self.time_head = nn.Linear(d_model, 3 * n_mix)   # LogNormal mixture: weights, means, scales
            self.mark_head = Predictor(d_model, num_types)   # same activity head as baseline_act (2026-10-26)
            if model_type == "mixture_joint":
                # Start at A3's independence submodel, with identical shared weights and CPU RNG.
                for name in ("tie_head", "time_head"):
                    head = getattr(self, name)
                    with torch.random.fork_rng(devices=[]):
                        expanded = nn.Linear(d_model, num_types * head.out_features)
                    with torch.no_grad():
                        expanded.weight.copy_(head.weight.repeat(num_types, 1))
                        expanded.bias.copy_(head.bias.repeat(num_types))
                    setattr(self, name, expanded)

    @property
    def config(self) -> dict:
        """A copy of the architecture settings; edit a copy before constructing."""
        return self._config.copy()

    @classmethod
    def from_config(cls, config: dict, *, seed: int | None = None):
        """Construct from exact kwargs; an explicit CPU init seed preserves RNG.

        With seed=None, construction consumes the usual CPU random stream.
        Seeded construction neither consumes it nor reseeds accelerators.
        """
        with torch.random.fork_rng(devices=[], enabled=seed is not None), torch.device("cpu"):
            if seed is not None:
                torch.random.default_generator.manual_seed(seed)
            return cls(**config)

    def checkpoint(self) -> dict:
        """Frozen CPU inference snapshot; optimizer/RNG state belongs to the trainer."""
        return {"model_config": self.config,
                "model": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}}

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, *, device="cpu"):
        """Reconstruct architecture and weights strictly, preserving caller RNG.

        For legacy weights, supply {"model_config": exact_kwargs, "model": state_dict}.
        Shapes alone cannot determine dropout or attention head count.
        """
        state = checkpoint["model"]
        model = cls.from_config(checkpoint["model_config"], seed=0)
        model.to(device=device, dtype=next(iter(state.values())).dtype)
        model.load_state_dict(state, strict=True)
        return model

    def forward(self, event_type, event_time):
        """Encoder state and heads: (time_pred, mark_logit) for baseline_act, (tie_logit, time_params, mark_logit) otherwise."""
        non_pad_mask = get_non_pad_mask(event_type)
        h = self.encoder(event_type, event_time, non_pad_mask)

        if self.model_type == "baseline_act":
            return h, (self.time_predictor(h, non_pad_mask), self.mark_head(h, non_pad_mask))

        tie_logit = self.tie_head(h)                          # [B, S, 1] or [B, S, 3]
        if self.model_type == "mixture":
            tie_logit = tie_logit.squeeze(-1)                 # [B, S]
        time_params = self.time_head(h)
        if self.model_type == "mixture_joint":
            time_params = time_params.view(*h.shape[:-1], self.num_types, 3 * self.n_mix)
        return h, (tie_logit, time_params, self.mark_head(h, non_pad_mask))
