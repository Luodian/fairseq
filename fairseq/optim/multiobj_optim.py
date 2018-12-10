import torch
from torch.optim.optimizer import Optimizer, required


def normalize_param(W):
    return W / W.norm(2).clamp(min=1e-12)


class MultiObjSGD(Optimizer):
    """
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, always_project=True):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, frozen=False)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(MultiObjSGD, self).__init__(params, defaults)
        self.always_project = always_project

    def __setstate__(self, state):
        super(MultiObjSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def save_constraints(self):

        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                # skip frozen parameters
                if getattr(param_state, "frozen", False):
                    continue
                param_state["constraint"] = torch.zeros_like(p.data)
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state["constraint"].add_(d_p)

    def apply_constraint(self, g_p, c_p):
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
                # skip frozen parameters
                if getattr(param_state, "frozen", False):
                    continue
                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(
                            p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if "constraint" in param_state:
                    d_p = self.apply_constraint(
                        d_p, param_state["constraint"])

                p.data.add_(-group["lr"], d_p)

        return loss


class AvgMultiObjSGD(MultiObjSGD):

    def apply_constraint(self, g_p, c_p):
        avg_p = 0.5 * (c_p + g_p)
        return avg_p


class OrthoMultiObjSGD(MultiObjSGD):

    def apply_constraint(self, g_p, c_p):
        c_unit = c_p / (c_p.norm(2) + 1e-10)
        dot = (g_p * c_unit).sum()
        if self.always_project or dot.data <= 0:
            return g_p - dot * c_unit
        else:
            return g_p


class AvgOrthoMultiObjSGD(MultiObjSGD):

    def apply_constraint(self, g_p, c_p):
        g_norm = (g_p.norm(2)+1e-10)
        c_norm = c_p.norm(2)+1e-10
        cosine = ((g_p / g_norm) * (c_p / c_norm)).sum()
        norm_ratio = g_norm / c_norm
        if self.always_project or cosine.data <= 0:
            return 0.5 * (g_p + c_p) - 0.5 * cosine * (norm_ratio * c_p + (1.0 / norm_ratio) * g_p)
        else:
            # If the two are somewhat aligned, no need to project
            return g_p


class FullOrthoMultiObjSGD(MultiObjSGD):

    def apply_constraint(self, g_p, c_p, dot_val):
        if False and dot_val > 0:
            # If the two are somewhat aligned, no need to project
            return g_p
        else:
            # Otherwise project
            return g_p - dot_val * c_p

    def compute_dot(self):
        dot_val = 0
        constraint_norm = 0
        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                param_state = self.state[p]
                # skip frozen parameters
                if getattr(param_state, "frozen", False):
                    continue
                constraint_norm += (param_state["constraint"]
                                    * param_state["constraint"]).sum().data
                dot_val += (d_p * param_state["constraint"]).sum().data
        dot_val = dot_val / (constraint_norm + 1e-20)
        return dot_val

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        dot_val = self.compute_dot()

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
                    continue
                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(
                            p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if "constraint" in param_state:
                    d_p = self.apply_constraint(
                        d_p, param_state["constraint"], dot_val)

                p.data.add_(-group["lr"], d_p)

        return loss


class CwiseOrthoMultiObjSGD(MultiObjSGD):

    def apply_constraint(self, g_p, c_p):
        mask = torch.nn.functional.relu(torch.sign(g_p * c_p))
        return mask * g_p


def build_optimizer(name, params, **kwargs):
    if name == "sgd":
        return MultiObjSGD(params, **kwargs)
    elif name == "avg":
        return AvgMultiObjSGD(params, **kwargs)
    elif name == "ortho":
        kwargs["normalize_constraint"] = True
        return OrthoMultiObjSGD(params, **kwargs)
    else:
        ValueError(f"Unknown optimizer {name}")
