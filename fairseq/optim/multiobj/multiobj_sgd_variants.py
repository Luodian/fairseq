import torch as th
from . import register_multiobj_optim
from .multiobj_sgd_base import MultiObjSGD


@register_multiobj_optim("single")
class SingleObjSGD(MultiObjSGD):
    """Same as SGD ("single" objective)"""

    def combine_gradients(self, g_p, aux_g_p):
        return g_p


@register_multiobj_optim("avg")
class AvgMultiObjSGD(MultiObjSGD):
    """Average the gradients"""

    def combine_gradients(self, g_p, aux_g_p):
        avg_p = 0.5 * (aux_g_p + g_p)
        return avg_p


@register_multiobj_optim("ortho")
class OrthoMultiObjSGD(MultiObjSGD):
    """Project the gradient g_p on the hyperplane orthogonal to aux_g_p"""

    def combine_gradients(self, g_p, aux_g_p):
        c_unit = aux_g_p / (aux_g_p.norm(2) + 1e-10)
        dot = (g_p * c_unit).sum()
        # Only project if the gradients have negative dot product
        if self.always_project or dot.data <= 0:
            return g_p - dot * c_unit
        else:
            return g_p


@register_multiobj_optim("nullify")
class NullifyMultiObjSGD(MultiObjSGD):
    """Nullify the gradient if the directions are not aligned"""

    def combine_gradients(self, g_p, aux_g_p):
        if (g_p * aux_g_p).sum() <= 0:
            return th.zeros_like(g_p)
        else:
            return aux_g_p


@register_multiobj_optim("cwise-ortho")
class CwiseOrthoMultiObjSGD(MultiObjSGD):
    """Orthogonal projection but at the level of scalar parameters"""

    def combine_gradients(self, g_p, aux_g_p):
        mask = th.nn.functional.relu(th.sign(g_p * aux_g_p))
        return mask * g_p


@register_multiobj_optim("cosine-weighted")
class CosineWeightedMultiObjSGD(MultiObjSGD):
    """Weight the update by the (rectified) cosine similarity between the two
    gradients. Update in the direction of aux_g_p"""

    def combine_gradients(self, g_p, aux_g_p):
        c_unit = aux_g_p / (aux_g_p.norm(2) + 1e-10)
        g_unit = g_p / (g_p.norm(2) + 1e-10)
        cosine = (g_unit * c_unit).sum()
        return th.nn.functional.relu(cosine) * g_p


@register_multiobj_optim("cosine-weighted-sum")
class CosineWeightedSumMultiObjSGD(MultiObjSGD):
    """Weight the update by the (rectified) cosine similarity between the two
    gradients. Update in the direction of g_p + aux_g_p
    (see https://arxiv.org/abs/1812.02224)"""

    def combine_gradients(self, g_p, aux_g_p):
        c_unit = aux_g_p / (aux_g_p.norm(2) + 1e-10)
        g_unit = g_p / (g_p.norm(2) + 1e-10)
        cosine = (g_unit * c_unit).sum()
        return th.nn.functional.relu(cosine) * 0.5 * (g_p + aux_g_p)


@register_multiobj_optim("colinear")
class ColinearMultiObjSGD(MultiObjSGD):
    """Project g_p on the direction of aux_g_p (when the 2 are colinear)"""

    def combine_gradients(self, g_p, aux_g_p):
        c_unit = aux_g_p / (aux_g_p.norm(2) + 1e-10)
        dot = (c_unit * g_p).sum()
        return th.nn.functional.relu(dot) * c_unit


@register_multiobj_optim("same-contrib")
class SameContribMultiObjSGD(MultiObjSGD):
    """Here the update is a vector d such that
    Loss_1(x + d) - Loss_1(x) = Loss_2(x + d) - Loss_2(x)"""

    def combine_gradients(self, g_p, aux_g_p):
        diff = g_p - aux_g_p
        diff_norm = diff.norm(2) + 1e-10
        diff_unit = diff / diff_norm
        dot = (g_p * diff_unit).sum()
        return g_p - dot * diff_unit


@register_multiobj_optim("avg-ortho")
class AvgOrthoMultiObjSGD(MultiObjSGD):
    """Project g_p on the orthogonal of aux_g_p, and aux_g_p on the orthogonal
    of g_p, then average"""

    def combine_gradients(self, g_p, aux_g_p):
        g_norm = g_p.norm(2)+1e-10
        c_norm = aux_g_p.norm(2)+1e-10
        dot = (g_p * aux_g_p).sum()
        if self.always_project or dot.data <= 0:
            g_unit = g_p / g_norm
            c_unit = aux_g_p / c_norm
            g_proj_c = g_p - (g_p * c_unit).sum() * c_unit
            aux_g_proj_g = aux_g_p - (aux_g_p * g_unit).sum() * g_unit
            return 0.5 * (g_proj_c + aux_g_proj_g)
        else:
            # If the two are somewhat aligned, no need to project
            return g_p
