# coding: utf-8
# implementation of numerical gradient

import numpy as np


def neumerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        # calc f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # calc f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad
