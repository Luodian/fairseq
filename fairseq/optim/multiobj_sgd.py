from . import FairseqOptimizer, register_optimizer

from .multiobj_optim import AvgMultiObjSGD, OrthoMultiObjSGD


@register_optimizer('multiobj_sgd')
class MultiObjSGD(FairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args, params)
        name = getattr(args, "multiobj_optim_name", "avg")
        if name == "avg":
            self._optimizer = AvgMultiObjSGD(params, **self.optimizer_config)
        elif name == "ortho":
            self.optimizer_config["normalize_constraint"] = True
            self._optimizer = OrthoMultiObjSGD(params, **self.optimizer_config)
        else:
            ValueError(f"Unknown optimizer {name}")
    
    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        parser.add_argument('--multiobj-optim-name', default='avg', metavar='NAME')

    def save_constraints(self):
        self.optimizer.save_constraints()

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
            'momentum': self.args.momentum,
            'weight_decay': self.args.weight_decay,
        }
