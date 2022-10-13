"""Module to create an adapted mesh.
# TODO: comments. """
import numpy as np
from scipy.optimize import fsolve


def mesh1D_adapted(r1, r2, nx):
    """1D adapted mesh

    :param r1: ??
    :param r2: ??
    :param nx: ??
    :return: p, t
    """
    dlay = 1
    dwall = 1e-2

    # Calculate mesh ratio
    c = 1 - dlay / dwall
    # TODO: what is this 1.1?
    rat = fsolve(lambda x_val: scalingfun(x_val, nx, c), 1.1)
    rat = rat[0]

    xv = np.zeros(nx + 1)
    xv[1] = dwall
    for i in range(1, nx):
        xv[i + 1] = xv[i] + dwall * rat ** i

    p = np.array([xv * (r2 - r1) + r1])
    t = np.array([[k, k + 1] for k in range(0, nx)])
    t = t.transpose()
    return p, t


def scalingfun(x, n, c):
    """??

    :param x: ??
    :param n: ??
    :param c: ??
    :return: ??
    """
    F = c

    for i in range(1, n):
        F += x ** i

    return F
