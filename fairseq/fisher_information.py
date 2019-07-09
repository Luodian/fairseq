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
        fim = th.load(precomputed)
        loaded_params = set([name for name in fim])
        orig_params = set([name for name in trainer.model.named_parameters()])
        if len(set(loaded_params) & set(orig_params)) < len(orig_params):
            raise ValueError("Parameters mismatch between model and FIM "
                             f"loaded from {precomputed}")
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
    # Initialize the Fisher
    FIM = {
        name: th.zeros_like(p)
        for name, p in trainer.model.named_parameters()
        if p.requires_grad
    }
    # Iterate
    for i, samples in enumerate(islice(progress, n_steps)):
        # Forward backward
        trainer.train_step(samples, update_params=False, clip_grad=False)
        # Get gradients
        for name, p in trainer.model.named_parameters():
            FIM[name].add_(p.grad.detach()**2)
        # Log progress
        if i % 10 == 0:
            progress.log({"step": i})
    # Normalize
    FIM = {name: F/n_steps for name, F in FIM.items()}
    return FIM


def inverse_fisher_rotate(params, fim):
    # Rescale gradients
    grads = {name: p.grad / (fim[name]+1e-12) for name, p in params}
    # compute old and new gradient norms
    grads_norm = th.cat([p.grad for _, p in params]).norm(2)
    new_norm = th.cat([p.grad for p in grads.values()]).norm(2)
    # Rescale so that the new gradient has the same norm
    ratio = (grads_norm / new_norm).item()
    grads = {name: g * ratio for name, g in grads.items()}
    # Replace gradients
    for name, p in params:
        p.grad.copy_(grads[name])
