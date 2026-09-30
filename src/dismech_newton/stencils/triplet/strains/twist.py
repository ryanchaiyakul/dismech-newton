"""Twist strain ``tau = theta_f - theta_e + reference twist`` (gradient, Hessian)."""

import warp as wp

from ....frames import skew
from .geometry import TripletGeometry
from .helper import add_edge_hessian, edge_gradient, mat11f, vec11f


@wp.func
def dtau(g: TripletGeometry) -> tuple[vec11f, mat11f]:
    """Theta enters linearly; the position part is the reference-twist derivative."""
    te = g.te
    tf = g.tf
    ne = g.ne
    nf = g.nf
    kb = g.kb

    J = edge_gradient(0.5 * kb / ne, 0.5 * kb / nf, -1.0, 1.0)

    te_tt = te + g.tt
    tf_tt = tf + g.tt
    Dee = -(wp.outer(kb, te_tt) + wp.outer(te_tt, kb)) / (4.0 * ne * ne)
    Dff = -(wp.outer(kb, tf_tt) + wp.outer(tf_tt, kb)) / (4.0 * nf * nf)
    Def = (2.0 / g.chi * skew(te) - wp.outer(kb, g.tt)) / (2.0 * ne * nf)
    H = add_edge_hessian(mat11f(), Dee, Def, Dff)
    return J, H
