import torch as th
from torch.optim.optimizer import Optimizer, required


def normalize_param(W):
    return W / W.norm(2).clamp(min=1e-12)


def to_vector(tensors):
    """Flatten a list of parameters/gradients to a vector"""
    return th.cat([t.view(-1) for t in tensors]).detach()


def from_vector(tensors, vector):
    """Reverse `to_vector` (overwrites the tensor values)"""
    pointer = 0
    for tensor in tensors:
        new_val = vector[pointer:pointer+tensor.numel()].view(tensor.size())
        tensor.copy_(new_val)
        pointer += tensor.numel()


class MultiObjSGD(Optimizer):
    """
    This optimizer works like SGD excepts:

    1. it stores gradient from an auxiliary task with `.save_auxiliary()`
    2. it uses those auxiliary gradients using `.combine_gradientss()` before
        applying the update

    Args:
        full_gradients (bool): do gradient combination ops on the full
            gradients (as opposed to separately for each parameter)
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, always_project=True,
                 full_gradients=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        frozen=False)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(MultiObjSGD, self).__init__(params, defaults)
        self.always_project = always_project
        self.full_gradients = full_gradients

    def __setstate__(self, state):
        super(MultiObjSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def save_auxiliary(self):
        """This saves the gradients wrt. the auxiliary objective"""

        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                # skip frozen parameters (TODO: remove this)
                if getattr(param_state, "frozen", False):
                    continue
                # Actually save the gradient
                param_state["aux_grad"] = th.zeros_like(p.data)
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state["aux_grad"].add_(d_p)

    def combine_gradients(self, g_p, aux_g_p):
        """Manipulate the gradient g_p using the gradient from the auxiliary
        objective aux_g_p"""
        raise NotImplementedError()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Apply momentum and everything to get final gradient
        params = []
        lrs = []
        grads = []
        aux_grads = []
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
                # skip frozen parameters
                if getattr(param_state, "frozen", False):
                    print("Skipping parameter of size", p.dim())
                    continue
                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = th.zeros_like(
                            p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                # Track parameters, learning rate, gradients and auxiliary
                # gradients
                params.append(p)
                lrs.append(group["lr"])
                grads.append(d_p)
                if "aux_grad" in param_state:
                    aux_grads.append(param_state["aux_grad"])
                else:
                    aux_grads.append(th.zeros_like(d_p))

        # Combine gradients
        if self.full_gradients:
            # Consider parameters as one vector
            new_grad_vec = self.combine_gradients(
                to_vector(grads),
                to_vector(aux_grads)
            )
            # Overwrite gradients
            from_vector(grads, new_grad_vec)
        else:
            # Treat each parameter independently
            grads = [self.combine_gradients(g, aux_g)
                     for g, aux_g in zip(grads, aux_grads)]

        # Apply the update
        for p, lr, g in zip(params, lrs, grads):
            p.data.add_(-lr, g)

        return loss

