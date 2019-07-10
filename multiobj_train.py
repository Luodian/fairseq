#!/usr/bin/env python3
import sys
import collections
import math
import torch

from fairseq import options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from fairseq.fisher_information import estimate_diagonal_fisher
from train import (
    checkpoint_utils,
    validate,
    get_training_stats,
)


def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)
    if init_distributed:
        raise ValueError("Distibuted training not supported by multiobj "
                         "training")

    # Print args
    print(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest
    # checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    if args.restore_file is not None:
        # Load from checkpoint
        print('| loading model from {}'.format(args.restore_file))
        [model], _model_args = checkpoint_utils.load_model_ensemble(
            [args.restore_file],
            arg_overrides=eval(args.model_overrides),
            task=task,
        )
        # Overwrite architecture arguments
        # (this is very hacky but I don't know a better way)
        for k, v in _model_args.__dict__.items():
            is_model_argument = k == "arch"
            is_model_argument |= k.startswith("encoder_")
            is_model_argument |= k.startswith("decoder_")
            is_model_argument |= k.startswith("share_")
            is_model_argument |= k.startswith("adaptive_")
            if hasattr(args, k) and is_model_argument:
                setattr(args, k, v)
    else:
        # Or build model from scratch
        model = task.build_model(args)

    # Training criterion
    criterion = task.build_criterion(args)
    print(model)
    print('| model {}, criterion {}'.format(
        args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Load auxiliary data
    epoch_aux_itr = task.get_batch_iterator(
        dataset=task.dataset(args.train_subset, idx=1),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            trainer.model.max_positions(),
        ),
        ignore_invalid_inputs=True,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        num_workers=args.num_workers,
        epoch=0,
    )

    # Estimate fisher if needed
    if args.inverse_fisher or args.ewc > 0:
        fisher_itr = task.get_batch_iterator(
            dataset=task.dataset(args.train_subset, idx=1),
            max_tokens=args.max_tokens,
            max_sentences=1,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.model.max_positions(),
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
            epoch=0,
        )
        fim = estimate_diagonal_fisher(
            args,
            trainer,
            fisher_itr,
            args.n_fisher_samples,
            precomputed=args.precomputed_fisher
        )
        trainer.fim = fim
    # EWC
    if args.ewc > 0.0:
        trainer.prepare_ewc(args.ewc)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr, epoch_aux_itr)

        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(
                args, trainer, task, epoch_itr, valid_subsets)
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, None)

        if ':' in getattr(args, 'data', ''):
            # sharded data: get train iterator for next epoch
            epoch_itr = trainer.get_train_iterator(epoch_itr.epoch)
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def print_gpu_stats():
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    print(f"Cached: {torch.cuda.memory_cached()/1e9:.1f}GB")


def train(args, trainer, task, epoch_itr, epoch_aux_itr, fim=None):
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]
    print(update_freq)
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    # Auxiliary iterator
    aux_itr = epoch_aux_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus)
    aux_itr = iterators.GroupedIterator(
        aux_itr, update_freq, bottomless=True)

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        # Record gradients from auxiliary data
        aux_samples = next(aux_itr)
        trainer.train_step(aux_samples, update_params=False)
        # Fisher
        if hasattr(trainer.optimizer, "save_auxiliary"):
            trainer.optimizer.save_auxiliary()
        else:
            print("Warning, the optimizer is ignoring the auxiliary gradients")
        # Take a step on the primary task
        log_output = trainer.train_step(
            samples,
            apply_ewc=args.ewc > 0
        )

        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses = validate(
                args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, None)

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def add_multiobj_args(parser):
    mto_group = parser.add_argument_group("Multi-objective related arguments")
    mto_group.add_argument("--async-save", action="store_true",
                           help="Save to ymp dir and async copy (much faster)")
    mto_group.add_argument("--save-only-model", action="store_true",
                           help="Don't save the optimizer history")
    mto_group.add_argument("--freeze-embeddings", action="store_true",
                           help="Freeze word embeddings when finetuning")
    mto_group.add_argument("--freeze-decoder", action="store_true",
                           help="Freeze decoder when finetuning")
    mto_group.add_argument("--inverse-fisher", action="store_true",
                           help="Multiply gradients by the inverse diagonal"
                           " empirical fisher information matrix")
    mto_group.add_argument("--n-fisher-samples", type=int, default=100,
                           help="Number of samples to estimate the Fisher "
                           "matrix")
    mto_group.add_argument("--precomputed-fisher", type=str,
                           help="Cache the Fisher to a file")
    mto_group.add_argument("--ewc", type=float, default=0.0,
                           help="Add elastic weight consolidation")
    mto_group.add_argument('--model-overrides', default="{}", type=str, metavar='DICT',
                           help='a dictionary used to override model args at generation '
                           'that were used during model training')


def cli_main():
    # Horrible hack, please close your eyes and don't look
    cli_args = set(sys.argv)
    print("Command line argumetns")
    print(cli_args)
    if "--arch" not in cli_args and "-a" not in cli_args:
        sys.argv.append("--arch")
        sys.argv.append("transformer_iwslt_de_en")
    print(cli_args)
    # It's over now you can look
    parser = options.get_training_parser()
    add_multiobj_args(parser)
    args = options.parse_args_and_arch(parser)

    if args.distributed_port > 0 or args.distributed_init_method is not None:
        raise NotImplementedError(
            "Multitask doesn't support multiprocessing yet")
        from distributed_train import main as distributed_main

        distributed_main(args)
    elif args.distributed_world_size > 1:
        raise NotImplementedError(
            "Multitask doesn't support multiprocessing yet")
        from multiprocessing_train import main as multiprocessing_main

        multiprocessing_main(args)
    else:
        main(args)


if __name__ == '__main__':
    cli_main()
