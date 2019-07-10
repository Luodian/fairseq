#!/usr/bin/env python3
"""
Functions for estimating and using the (empirical) Fisher Information Matrix
"""
import os.path
import torch as th
from itertools import islice

from fairseq.data import iterators
from fairseq import progress_bar


def estimate_diagonal_fisher(
    args,
    trainer,
    epoch_itr,
    n_steps,
    precomputed=None
):
    if precomputed is not None and os.path.isfile(precomputed):
        print(f"| Loading precomputed FIM from {precomputed}")
        fim = th.load(precomputed)
        loaded_params = [name for name in fim]
        orig_params = [name for name, _ in trainer.model.named_parameters()]
        # Check that the two parameter sets are the same
        not_in_model = set(loaded_params) - set(orig_params)
        not_in_fim = set(orig_params) - set(loaded_params)
        if len(not_in_model) > 0:
            raise ValueError(
                "| ERROR: Parameters mismatch between model and FIM loaded"
                f" from {precomputed}: the following parameters are in the"
                f" FIM but not in the model {list(not_in_model)}"
            )
        elif len(not_in_fim) > 0:
            raise ValueError(
                "| ERROR: Parameters mismatch between model and FIM loaded"
                f" from {precomputed}: the following parameters are in the"
                f" model but not in the FIM {list(not_in_fim)}"
            )
        return fim
    else:
        fim = _estimate_diagonal_fisher(args, trainer, epoch_itr, n_steps)
        if precomputed is not None:
            th.save(fim, precomputed)
        return fim


def _estimate_diagonal_fisher(args, trainer, epoch_itr, n_steps):
    """Estimate the diagonal empirical fisher information matrix"""
    # Iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=True
    )
    itr = iterators.GroupedIterator(itr, 1, bottomless=True)
    progress = progress_bar.build_progress_bar(
        args, itr, 0, no_progress_bar='simple',
    )
    progress.log_interval = n_steps // 10
    # Initialize the Fisher
    FIM = {
        name: th.zeros_like(p)
        for name, p in trainer.model.named_parameters()
    }
    # Iterate
    for i, samples in enumerate(islice(progress, n_steps)):
        # Forward backward
        trainer.train_step(samples, update_params=False, clip_grad=False)
        # Get gradients
        for name, p in trainer.model.named_parameters():
            FIM[name].add_(p.grad.detach()**2)
        # Log progress
        progress.log({"step": i})
    # Normalize
    FIM = {name: F/n_steps for name, F in FIM.items()}
    return FIM


def inverse_fisher_rotate(params, fim):
    """Multiply by the inverse diagonal FIM and renormalize"""
    params = {name: p for name, p in params}
    # Rescale gradients
    grads = {name: p.grad / (fim[name]+1e-12) for name, p in params.items()}
    # compute old and new gradient norms
    grads_norm = th.cat([p.grad.view(-1) for p in params.values()]).norm(2)
    new_norm = th.cat([g.view(-1) for g in grads.values()]).norm(2)
    # Rescale so that the new gradient has the same norm
    ratio = (grads_norm / new_norm).item()
    grads = {name: g * ratio for name, g in grads.items()}
    # Replace gradients
    for name, p in params.items():
        p.grad.copy_(grads[name])


def ewc_loss(params, orig_params, fim, epsilon):
    """Compute the EWC loss. In particular the fisher is rescaled so that
    its average diagonal value is 1 (to make the scaling more similar to
    vanilla l2 reg)"""
    params = {name: p for name, p in params}
    param_names = [name for name in params]
    # No FIM? just standard L2 then
    if fim is None:
        fim = {name: th.ones_like(p) for name, p in params.items()}
    # Trace of the FIM
    fim_trace = sum(f.sum().item() for f in fim.values())
    # Total number of params
    n_params = sum(f.numel() for f in fim.values())
    # the total weight is epsilon * |params| / fim_trace
    # this is so that the average penalty for each weight has the same
    # magnitude as for standard L2 (helps choose epsilon independently
    # from the magnituted of the FIM)
    weight = epsilon * n_params / fim_trace
    # Total penalty
    penalty = weight * th.sum(th.stack([
        th.sum(fim[name] * (params[name] - orig_params[name].detach()) ** 2)
        for name in param_names
    ]))
    return penalty
