"""Losses of plan.md C2, summed over the masked next-event targets.

Shapes: B batch, S sequence length, T activities, K mixture components. Position i of a model output
predicts event i + 1. Target-aligned inputs (types, dt, R, cause, target_mask) are indexed by the target
event, so outputs use [:, :-1] and targets [:, 1:]. Activities are 1-based (0 = PAD). Δt in days.
"""
import math

import torch
import torch.nn.functional as F


def ce_nll(logits, types, target_mask):
    """Cross-entropy on the logged next activity (A0, A1); logits [B,S,T], types [B,S], target_mask [B,S] bool."""
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    y = (types[:, 1:] - 1).clamp(min=0)
    nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return (nll * target_mask[:, 1:]).sum()


def set_nll(logits, R, target_mask):
    """Set likelihood −log Σ_{a∈R} p(a | h) (A2–A4); R [B,S,T] bool = remaining block members of each target; equals CE when |R| = 1."""
    m = target_mask[:, 1:]
    R = R[:, 1:] | ~m.unsqueeze(-1)                      # unmasked rows: any non-empty set, keeps the loss finite
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    nll = -torch.logsumexp(logp.masked_fill(~R, float("-inf")), dim=-1)
    return (nll * m).sum()


def block_set_nll(logits, R, target_mask, dt):
    """A3M (design b′, plan.md §2 C2): set likelihood on targets with Δt > 0 only (block starts, untied); Δt = 0 targets have no activity loss."""
    return set_nll(logits, R, target_mask & (dt > 0))


def block_start(dt):
    """[B,S] -> [B,S]: position of the first event of the tie block holding each event (last position ≤ it with Δt > 0, else 0)."""
    pos = torch.arange(dt.size(1), device=dt.device).expand_as(dt)
    return torch.where(dt > 0, pos, torch.zeros_like(pos)).cummax(dim=1).values


def pre_block_index(dt):
    """[B,S] -> [B,S]: position of the state just before each event's tie block (block start − 1, ≥ 0); events of the first block use position 0."""
    return (block_start(dt) - 1).clamp(min=0)


def pre_block_set_nll(logits, R, target_mask, dt):
    """A3B (design b, plan.md §2 C2): set likelihood of each target from the state before its tie block; untied targets as A3."""
    idx = pre_block_index(dt)[:, 1:].unsqueeze(-1).expand(-1, -1, logits.size(-1))
    shifted = torch.cat([logits.gather(1, idx), logits[:, -1:]], dim=1)   # position i holds the logits for target i + 1
    return set_nll(shifted, R, target_mask)


def risk_set_nll(logits, R, target_mask, types, risk):
    """A3C (control of plan_exp.md 2.10): CE on the logged label at at-risk targets (risk [B,S] bool, train-fixed), set likelihood elsewhere."""
    return set_nll(logits, R, target_mask & ~risk) + ce_nll(logits, types, target_mask & risk)


def _log_mixture(params, dt, K):
    """log Σ_k w_k LogNormal(dt; μ_k, σ_k) for dt > 0 (THP-M); entries with dt = 0 get a finite placeholder."""
    mu = params[..., K:2 * K]
    sigma = F.softplus(params[..., 2 * K:]) + 1e-4
    log_w = F.log_softmax(params[..., :K], dim=-1)
    log_dt = torch.log(dt.clamp(min=1e-8)).unsqueeze(-1)
    log_pdf = -log_dt - torch.log(sigma) - 0.5 * math.log(2 * math.pi) - 0.5 * ((log_dt - mu) / sigma) ** 2
    return torch.logsumexp(log_w + log_pdf, dim=-1)


def zero_atom_nll(tie_logit, time_params, dt, target_mask, K, cause=None):
    """Zero-inflated LogNormal mixture NLL; A1–A3: tie_logit [B,S], −log π₀ or −log(1−π₀) − log f; A4: tie_logit [B,S,3], cause [B,S] in {0 date, 1 batch, 2 positive}, −log p(cause) − 𝟙(Δt>0) log f."""
    m = target_mask[:, 1:]
    dt = dt[:, 1:]
    pos = (dt > 0).float()
    log_f = _log_mixture(time_params[:, :-1], dt, K)
    if tie_logit.dim() == 2:
        pi_logit = tie_logit[:, :-1]
        nll = (1 - pos) * -F.logsigmoid(pi_logit) + pos * (-F.logsigmoid(-pi_logit) - log_f)
    else:
        assert cause is not None, "A4 needs the cause of each target"
        c = cause[:, 1:].clamp(min=0)
        assert bool(((c == 2) == (dt > 0))[m].all()), "cause 2 (positive) must match Δt > 0"
        logp = F.log_softmax(tie_logit[:, :-1], dim=-1)
        nll = -logp.gather(-1, c.unsqueeze(-1)).squeeze(-1) - pos * log_f
    return (nll * m).sum()


def joint_set_nll(logits, tie_logit, time_params, dt, R, target_mask, K):
    """A3J: −log Σ_{a∈R} p(a|h) q(dt|a,h); tie [B,S,T], params [B,S,T,3K]."""
    m = target_mask[:, 1:]
    candidates = R[:, 1:] | ~m.unsqueeze(-1)
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    d = dt[:, 1:].unsqueeze(-1)
    pi = tie_logit[:, :-1]
    log_f = _log_mixture(time_params[:, :-1], d, K)
    log_q = torch.where(d > 0, F.logsigmoid(-pi) + log_f, F.logsigmoid(pi))
    nll = -torch.logsumexp((logp + log_q).masked_fill(~candidates, float("-inf")), dim=-1)
    return (nll * m).sum()


def eps_nll(time_params, dt, target_mask, K, eps):
    """A3-ε: LogNormal mixture NLL on max(Δt, ε), no zero atom (isolates the zero atom of A3)."""
    log_f = _log_mixture(time_params[:, :-1], dt[:, 1:].clamp(min=eps), K)
    return (-log_f * target_mask[:, 1:]).sum()


def _softplus(x, beta):
    """THP softplus with the argument capped at 20 (overflow guard)."""
    return torch.log1p(torch.exp((beta * x).clamp(max=20))) / beta


def thp_nll(model, h, event_time, types, time_pred, dt, target_mask, n_samples=100):
    """Original THP loss without its CE term: −(event LL − non-event LL) over all events + Σ (Δt̂ − Δt)² / 100 over the masked targets."""
    non_pad = types.ne(0).float()
    type_mask = F.one_hot(types, model.num_types + 1)[..., 1:].float()      # [B,S,T], PAD row = 0

    lam = _softplus(model.linear(h), model.beta)                             # [B,S,T]
    type_lam = (lam * type_mask).sum(-1)
    event_ll = (torch.log(type_lam + 1e-9) * non_pad).sum()

    gap = dt[:, 1:] * non_pad[:, 1:]                                         # Monte Carlo integral of the compensator
    u = gap.unsqueeze(2) * torch.rand(*gap.size(), n_samples, device=h.device)
    u = u / (event_time[:, :-1] + 1).unsqueeze(2)
    hid = (model.linear(h)[:, 1:] * type_mask[:, 1:]).sum(-1, keepdim=True)
    non_event_ll = (_softplus(hid + model.alpha * u, model.beta).mean(-1) * gap).sum()

    se = ((time_pred[:, :-1].squeeze(-1) - dt[:, 1:]) ** 2 * target_mask[:, 1:]).sum()
    return -(event_ll - non_event_ll) + se / 100.0
