import numpy as np

def trapezoidal_rule(f, a, b, n):

    x = np.linspace(a, b, n+1)
    y = f(x)

    h = (b - a) / n
    I = h * (y[0] + 2 * sum(y[1:-1]) + y[-1]) / 2

    return I