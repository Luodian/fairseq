import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required


def normalize_param(W):
    return W / W.norm(2).clamp(min=1e-12)


class MultiObjSGD(Optimizer):
    """
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, normalize_constraint=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        normalize_constraint=normalize_constraint)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MultiObjSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def save_constraints(self):

        for group in self.param_groups:
            normalize_constraint = group["normalize_constraint"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                param_state["constraint_normal"] = torch.zeros_like(p.data)
                if normalize_constraint:
                    param_state["constraint_normal"].add_(normalize_param(d_p))
                else:
                    param_state["constraint_normal"].add_(d_p)

    def apply_constraints(self, g_p, c_p):
        return g_p


    def step(self, closure=None):
            """Performs a single optimization step.

            Arguments:
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
            """
            loss = None
            if closure is not None:
                loss = closure()

            for group in self.param_groups:
                weight_decay = group["weight_decay"]
                momentum = group["momentum"]
                dampening = group["dampening"]
                nesterov = group["nesterov"]

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)

                    param_state = self.state[p]
                    if momentum != 0:
                        if "momentum_buffer" not in param_state:
                            buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                            buf.mul_(momentum).add_(d_p)
                        else:
                            buf = param_state["momentum_buffer"]
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    if "constraint_normal" in param_state:
                        d_p = self.apply_constraint(d_p, param_state["constraint_normal"])

                    p.data.add_(-group["lr"], d_p)

            return loss

class AvgMultiObjSGD(MultiObjSGD):

    def apply_constraint(self, g_p, c_p):
        return 0.5 * (c_p + g_p)


class OrthoMultiObjSGD(MultiObjSGD):

    def apply_constraint(self, g_p, c_p):
        dot = (g_p * c_p).sum()
        if dot.data > 0:
            # If the two are somewhat aligned, no need to project
            return g_p
        else:
            # Otherwise project
            return g_p - dot * c_p

def build_optimizer(name, params, **kwargs):
    if name=="sgd":
        return MultiObjSGD(params, **kwargs)
    elif name=="avg":
        return AvgMultiObjSGD(params, **kwargs)
    elif name=="ortho":
        kwargs["normalize_constraint"] = True
        return OrthoMultiObjSGD(params, **kwargs)
    else:
        ValueError(f"Unknown optimizer {name}")

