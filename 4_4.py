# coding: utf-8
# gradient descent

import numpy as np


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (h*2)

        x[idx] = tmp_val

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):

    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


def func_2(x):
    return x[0]**2 + x[1]**2


if __name__ == '__main__':

    init_x = np.array([-3.0, 4.0])
    x = gradient_descent(func_2, init_x=init_x, lr=0.1, step_num=100)
    print(x)
