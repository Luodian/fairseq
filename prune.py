#!/usr/bin/env python3
import collections
import torch

from fairseq import options, progress_bar, tasks, utils
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from train import (
    load_dataset_splits,
    load_checkpoint,
)
from itertools import islice


def main(args):
    if args.max_tokens is None:
        args.max_tokens = 6000
    print(args)

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load dataset splits
    load_dataset_splits(task, ['train'])

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print('| model {}, criterion {},'.format(
        args.arch, criterion.__class__.__name__))
    print('| num. model params: {}'.format(sum(p.numel()
                                               for p in model.parameters())))

    # Make a dummy batch to (i) warm the caching allocator and (ii) as a
    # placeholder DistributedDataParallel when there's an uneven number of
    # batches per worker.
    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        model.max_positions(),
    )
    dummy_batch = task.dataset('train').get_dummy_batch(
        args.max_tokens, max_positions)

    # Build trainer
    trainer = Trainer(args, task, model, criterion, dummy_batch)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))
    print('| Optimizer {}'.format(trainer.optimizer.__class__.__name__))

    # Initialize dataloader
    epoch_itr = task.get_batch_iterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
    )

    # Load the latest checkpoint if one is available
    if not load_checkpoint(args, trainer, epoch_itr):
        trainer.dummy_train_step([dummy_batch])

    # Train until the learning rate gets too small
    prune_meter = StopwatchMeter()
    prune_meter.start()
    # Estimate head importance scores
    head_importance = estimate_head_importance(args, trainer, task, epoch_itr)
    prune_meter.stop()
    print('| done estimating head importance in {:.1f} seconds'.format(prune_meter.sum))

    # Print
    print("Head importances")
    print("Encoder self attention")
    for layer in range(head_importance["encoder_self"].size(0)):
        print("\t".join(f"{x:.5f}" for x in head_importance["encoder_self"][layer]))
    print("Encoder decoder attention")
    for layer in range(head_importance["encoder_decoder"].size(0)):
        print("\t".join(f"{x:.5f}" for x in head_importance["encoder_decoder"][layer]))
    print("Decoder self attention")
    for layer in range(head_importance["decoder_self"].size(0)):
        print("\t".join(f"{x:.5f}" for x in head_importance["decoder_self"][layer]))
    # Print sorted pruning profile
    encoder_self_profile = get_profile(head_importance["encoder_self"], prefix="E")
    encoder_decoder_profile = get_profile(head_importance["encoder_decoder"], prefix="A")
    decoder_self_profile = get_profile(head_importance["decoder_self"], prefix="D")
    # Join all
    all_profiles = {}
    all_profiles.update(encoder_self_profile)
    all_profiles.update(encoder_decoder_profile)
    all_profiles.update(decoder_self_profile)
    sorted_profiles = sorted(all_profiles.items(), key=lambda x: x[1])
    print("Heads sorted by importance:")
    print(" ".join(p for p, _ in sorted_profiles))
    print("Sorted head importance scores:")
    print(" ".join(f"{v.data:.5f}" for _, v in sorted_profiles))


def get_profile(importances, prefix):
    n_layers, n_heads = importances.size()
    return {
        f"{prefix}:{layer}:{head}": importances[layer, head]
        for layer in range(n_layers)
        for head in range(n_heads)
    }


def batch_head_importance(attn_variables):
    # Retrieve context (shape bsz x nheads x L x dhead) and mask (shape bsz x L)
    ctx = attn_variables["context"]
    mask = attn_variables["mask"]
    # Reverse mask
    if mask is not None:
        mask = torch.eq(mask, 0.0).float()
    else:
        mask = torch.ones(ctx.size(0), ctx.size(2)).to(ctx.device)
    # Context gradient
    d_ctx = ctx.grad
    # Take the absolute dot
    importance = torch.einsum(
        "bhli,bhlj->bhl",
        [ctx, d_ctx],
    )
    importance *= mask.unsqueeze(1)
    importance = importance.abs().sum(-1).sum(0).detach()
    denom = mask.sum()
    return importance, denom


def estimate_head_importance(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus)
    if args.n_pruning_steps > 0:
        itr = islice(itr, args.n_pruning_steps)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )
    # Inititalize meters
    extra_meters = collections.defaultdict(lambda: AverageMeter())
    # Initialize head importance scores
    encoder_layers = trainer.args.encoder_layers
    decoder_layers = trainer.args.decoder_layers
    encoder_heads = trainer.args.encoder_attention_heads
    decoder_heads = trainer.args.decoder_attention_heads
    device = next(trainer.model.parameters()).device
    head_importance = {
        "encoder_self": torch.zeros(encoder_layers, encoder_heads).to(device),
        "encoder_decoder": torch.zeros(decoder_layers, decoder_heads).to(device),
        "decoder_self": torch.zeros(decoder_layers, decoder_heads).to(device),
    }
    # Denominators to normalize properly
    denoms = {attn_type: val.clone() for attn_type, val in head_importance.items()}
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        # Compute gradients
        log_output = trainer.prune_step(samples)
        # Retrieve importance scores for the encoder
        for layer in range(encoder_layers):
            self_attn_variables = trainer.model.encoder.layers[layer].self_attn_variables
            importance, denom = batch_head_importance(self_attn_variables)
            head_importance["encoder_self"][layer] += importance
            denoms["encoder_self"][layer] += denom
        # Retrieve importance scores for the decoder
        for layer in range(decoder_layers):
            # Self attention
            self_attn_variables = trainer.model.decoder.layers[layer].self_attn_variables
            importance, denom = batch_head_importance(self_attn_variables)
            head_importance["decoder_self"][layer] += importance
            denoms["decoder_self"][layer] += denom
            # Encoder attention
            encoder_attn_variables = trainer.model.decoder.layers[layer].encoder_attn_variables
            importance, denom = batch_head_importance(encoder_attn_variables)
            head_importance["encoder_decoder"][layer] += importance
            denoms["encoder_decoder"][layer] += denom
        # log mid-epoch stats
        stats = get_pruning_stats(trainer)
        for k, v in log_output.items():
            extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats)
        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()
    # log end-of-epoch stats
    stats = get_pruning_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)
    # Normalize by type
    for attn_type in denoms:
        head_importance[attn_type] /= denoms[attn_type]
    # Normalize by layer
    if args.normalize_by_layer:
        for attn_type, importance in head_importance.items():
            for layer in range(importance.size(0)):
                head_importance[attn_type][layer] /= torch.sqrt((importance[layer]**2).sum())
    return {k: v.cpu() for k, v in head_importance.items()}


def get_pruning_stats(trainer):
    stats = collections.OrderedDict()
    stats['wps'] = round(trainer.get_meter('wps').avg)
    stats['wpb'] = round(trainer.get_meter('wpb').avg)
    stats['bsz'] = round(trainer.get_meter('bsz').avg)
    stats['oom'] = trainer.get_meter('oom').avg
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = round(trainer.get_meter('train_wall').sum)
    return stats


def add_pruning_args(parser):
    group = parser.add_argument_group('Pruning')
    group.add_argument('--n-pruning-steps', default=0, type=int, metavar='N',
                       help='Number of steps to estimate the head importance scores')
    group.add_argument("--normalize-by-layer", action="store_true")


if __name__ == '__main__':
    parser = options.get_training_parser()
    add_pruning_args(parser)
    args = options.parse_args_and_arch(parser)

    if args.distributed_port > 0 or args.distributed_init_method is not None:
        raise NotImplementedError(
            "Pruning doesn't support multiprocessing yet")
        from distributed_train import main as distributed_main

        distributed_main(args)
    elif args.distributed_world_size > 1:
        raise NotImplementedError(
            "Pruning doesn't support multiprocessing yet")
        from multiprocessing_train import main as multiprocessing_main

        multiprocessing_main(args)
    else:
        main(args)
