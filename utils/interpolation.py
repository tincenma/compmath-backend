# interpolation.py

import numpy as np
from scipy.interpolate import CubicSpline, interp1d

def newton_forward_interpolation(x, y, x_value):
    """
    Newton's Forward Interpolation.
    
    Parameters:
        x: numpy array of equispaced data points.
        y: numpy array of corresponding function values.
        x_value: the point at which to interpolate.
        
    Returns:
        Interpolated value at x_value.
    """
    n = len(y)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = diff_table[i+1, j-1] - diff_table[i, j-1]
    h = x[1] - x[0]
    p = (x_value - x[0]) / h
    result = y[0]
    product = 1
    factorial = 1
    for i in range(1, n):
        product *= (p - (i - 1))
        factorial *= i
        result += (product * diff_table[0, i]) / factorial
    return result

def newton_backward_interpolation(x, y, x_value):
    """
    Newton's Backward Interpolation.
    
    Parameters:
        x: numpy array of equispaced data points.
        y: numpy array of corresponding function values.
        x_value: the point at which to interpolate.
        
    Returns:
        Interpolated value at x_value.
    """
    n = len(y)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y
    for j in range(1, n):
        for i in range(j, n):
            diff_table[i, j] = diff_table[i, j-1] - diff_table[i-1, j-1]
    h = x[1] - x[0]
    p = (x_value - x[-1]) / h
    result = y[-1]
    product = 1
    factorial = 1
    for i in range(1, n):
        product *= (p + (i - 1))
        factorial *= i
        result += (product * diff_table[-1, i]) / factorial
    return result

def central_difference_interpolation(x, y, x_value):
    """
    Central Difference Interpolation.
    
    Parameters:
        x: numpy array of equispaced data points.
        y: numpy array of corresponding function values.
        x_value: the point at which to interpolate.
        
    Returns:
        Interpolated value at x_value.
    """
    n = len(y)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = diff_table[i+1, j-1] - diff_table[i, j-1]
    h = x[1] - x[0]
    mid = n // 2
    p = (x_value - x[mid]) / h
    result = y[mid]
    factorial = 1
    product = 1
    # This implementation uses a simple alternating scheme.
    for i in range(1, n - mid):
        factorial *= i
        if i % 2 == 1:
            product *= (p - (i // 2))
            result += (product * diff_table[mid - (i // 2), i]) / factorial
        else:
            product *= (p + (i // 2))
            result += (product * diff_table[mid - (i // 2), i]) / factorial
    return result

def lagrange_interpolation(x, y, x_value):
    """
    Lagrange's Interpolation Formula.
    
    Parameters:
        x: list or numpy array of data points.
        y: list or numpy array of function values.
        x_value: the point at which to interpolate.
        
    Returns:
        Interpolated value at x_value.
    """
    n = len(x)
    result = 0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term *= (x_value - x[j]) / (x[i] - x[j])
        result += term
    return result

def cubic_spline_interpolation(x, y, x_value, kind='cubic'):
    """
    Cubic Spline Interpolation.
    
    Parameters:
        x: list or numpy array of data points.
        y: list or numpy array of function values.
        x_value: the point at which to interpolate.
        kind: type of interpolation; default is 'cubic'. For other kinds (e.g., 'quadratic'),
              interp1d from scipy.interpolate is used.
              
    Returns:
        Interpolated value at x_value.
    """
    if kind == 'cubic':
        cs = CubicSpline(x, y)
        return cs(x_value)
    else:
        f = interp1d(x, y, kind=kind)
        return f(x_value)

def newton_divided_difference_interpolation(x, y, x_value):
    """
    Newton's Divided Difference Interpolation.
    
    Parameters:
        x: list or numpy array of data points.
        y: list or numpy array of function values.
        x_value: the point at which to interpolate.
        
    Returns:
        Interpolated value at x_value.
    """
    n = len(x)
    div_diff = np.zeros((n, n))
    div_diff[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            div_diff[i, j] = (div_diff[i+1, j-1] - div_diff[i, j-1]) / (x[i+j] - x[i])
    result = div_diff[0, 0]
    product = 1
    for i in range(1, n):
        product *= (x_value - x[i-1])
        result += product * div_diff[0, i]
    return result