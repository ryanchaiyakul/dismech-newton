import warp as wp

from .helper import mat11f, vec11f


@wp.func
def deps(
    te: wp.vec3, tf: wp.vec3, ne: float, nf: float, l0e: float, l0f: float
) -> tuple[vec11f, vec11f, mat11f, mat11f]:
    Je = vec11f()
    Jf = vec11f()
    He = mat11f()
    Hf = mat11f()
    de = te / l0e
    df = tf / l0f
    Pe = (wp.identity(3, dtype=float) - wp.outer(te, te)) / (l0e * ne)
    Pf = (wp.identity(3, dtype=float) - wp.outer(tf, tf)) / (l0f * nf)
    for k in range(3):
        Je[k] = -de[k]
        Je[4 + k] = de[k]
        Jf[4 + k] = -df[k]
        Jf[8 + k] = df[k]
        for l in range(3):
            He[k, l] = Pe[k, l]
            He[k, 4 + l] = -Pe[k, l]
            He[4 + k, l] = -Pe[k, l]
            He[4 + k, 4 + l] = Pe[k, l]
            Hf[4 + k, 4 + l] = Pf[k, l]
            Hf[4 + k, 8 + l] = -Pf[k, l]
            Hf[8 + k, 4 + l] = -Pf[k, l]
            Hf[8 + k, 8 + l] = Pf[k, l]
    return Je, Jf, He, Hf
