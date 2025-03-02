# numerical_methods.py

import cmath
import numpy as np

# ----------------------- Root Finding Methods -----------------------

def bisection_method(f, a, b, tol=1e-6, return_iterations=False):
    """Bisection method for finding a root of f(x)=0 on [a, b]."""
    if f(a)*f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    iterations = 0
    midpoint = (a + b) / 2.0
    while abs(f(midpoint)) > tol:
        iterations += 1
        if f(a)*f(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
        midpoint = (a + b) / 2.0
    if return_iterations:
        return midpoint, iterations
    return midpoint

def newton_raphson_method(f, df, x0, tol=1e-6):
    """Newton-Raphson method for finding a root."""
    x = x0
    while abs(f(x)) > tol:
        x = x - f(x)/df(x)
    return x

def secant_method(f, x0, x1, tol=1e-6, return_iterations=False):
    """Secant method for finding a root."""
    iterations = 0
    while abs(f(x1)) > tol:
        iterations += 1
        x_temp = x1 - f(x1)*(x1 - x0)/(f(x1) - f(x0))
        x0, x1 = x1, x_temp
    if return_iterations:
        return x1, iterations
    return x1

def fixed_point_iteration(g, x0, tol=1e-6, max_iter=100):
    """Fixed point iteration method for solving x = g(x)."""
    x = x0
    for _ in range(max_iter):
        x_next = g(x)
        if abs(x_next - x) < tol:
            return x_next
        x = x_next
    raise Exception("Fixed point iteration did not converge.")

def false_position_method(f, a, b, tol=1e-6):
    """False Position (Regula Falsi) method for finding a root."""
    if f(a)*f(b) >= 0:
        raise ValueError("f(a) and f(b) must have opposite signs.")
    c = a
    while abs(f(c)) > tol:
        c = b - f(b)*(b - a)/(f(b) - f(a))
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
    return c

def muller_method(f, x0, x1, x2, tol=1e-6, max_iter=100):
    """Muller's method for finding a root (can find complex roots)."""
    for _ in range(max_iter):
        h0 = x1 - x0
        h1 = x2 - x1
        delta0 = (f(x1) - f(x0)) / h0
        delta1 = (f(x2) - f(x1)) / h1
        a = (delta1 - delta0) / (h1 + h0)
        b = a * h1 + delta1
        c = f(x2)

        discriminant = cmath.sqrt(b**2 - 4*a*c)
        if abs(b + discriminant) > abs(b - discriminant):
            denominator = b + discriminant
        else:
            denominator = b - discriminant
        x3 = x2 - (2*c)/denominator
        if abs(x3 - x2) < tol:
            return x3
        x0, x1, x2 = x1, x2, x3
    raise Exception("Muller's method did not converge.")

# ----------------------- Methods for Solving Systems of Equations -----------------------

def jacobi_method(A, b, x0, tol=1e-6, max_iter=100):
    """Jacobi iterative method for solving Ax = b."""
    n = len(b)
    x = x0.copy()
    for _ in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

def gauss_elimination(A, b):
    """Gauss elimination method for solving Ax = b."""
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    for i in range(n):
        max_row = np.argmax(abs(A[i:n, i])) + i
        A[[i, max_row]] = A[[max_row, i]]
        b[i], b[max_row] = b[max_row], b[i]
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x

def gauss_jordan(A, b):
    """Gauss-Jordan elimination for solving Ax = b."""
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    augmented = np.hstack((A, b.reshape(-1, 1)))
    for i in range(n):
        augmented[i] = augmented[i] / augmented[i, i]
        for j in range(n):
            if i != j:
                augmented[j] -= augmented[j, i] * augmented[i]
    return augmented[:, -1]

def gauss_seidel(A, b, x0, tol=1e-6, max_iter=100):
    """Gauss-Seidel method for solving Ax = b."""
    n = len(b)
    x = x0.copy()
    for _ in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

def relaxation_method(A, b, x0, tol=1e-6, max_iter=100, omega=1.2):
    """Successive over-relaxation (SOR) method for solving Ax = b."""
    n = len(b)
    x = x0.copy()
    for _ in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (1 - omega)*x[i] + omega*(b[i] - s) / A[i][i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

# ----------------------- Matrix Inversion and LU Factorization -----------------------

def iterative_matrix_inverse(A, tol=1e-6, max_iter=100):
    """
    Compute the inverse of matrix A using an iterative method.
    
    The initial guess B is chosen as the transpose of A divided by the product of its
    1-norm and infinity norm.
    """
    n = A.shape[0]
    I = np.eye(n)
    B = np.linalg.inv(A) + 0.1 * np.random.randn(3, 3)  # Noisy initial approximation
    for _ in range(max_iter):
        E = np.dot(A, B) - I  # Calculating the error
        B_new = B - np.dot(B, E)  # Proximity update
        if np.linalg.norm(E, ord='fro') < tol:
            return B_new
        B = B_new
    return B_new

def lu_factorization(A):
    """LU factorization of matrix A (A = LU)."""
    A = A.astype(float)
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    for i in range(n):
        for j in range(i+1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] -= factor * U[i, i:]
    return L, U

def solve_lu(L, U, b):
    """Solve Ax = b given LU factorization of A."""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

# ----------------------- Eigenvalue and QR Methods -----------------------

def power_method(A, v0, tol=1e-6, max_iter=100):
    """Power method to compute the dominant eigenvalue and eigenvector of A."""
    v = v0 / np.linalg.norm(v0)
    for _ in range(max_iter):
        w = A.dot(v)
        lambda_new = np.dot(w, v)
        v_new = w / np.linalg.norm(w)
        if np.linalg.norm(v_new - v) < tol:
            return lambda_new, v_new
        v = v_new
    return lambda_new, v_new

def jacobi_eigenvalues(A, tol=1e-6, max_iter=100):
    """
    Jacobi's method for computing all eigenvalues (and eigenvectors) of a symmetric matrix A.
    
    Returns a tuple (eigenvalues, eigenvectors).
    """
    A = A.copy()
    n = A.shape[0]
    V = np.eye(n)
    for _ in range(max_iter):
        max_val = 0
        p, q = 0, 1
        for i in range(n):
            for j in range(i+1, n):
                if abs(A[i, j]) > abs(max_val):
                    max_val = A[i, j]
                    p, q = i, j
        if abs(max_val) < tol:
            break
        theta = 0.5 * np.arctan2(2*A[p, q], A[p, p]-A[q, q])
        cos = np.cos(theta)
        sin = np.sin(theta)
        R = np.eye(n)
        R[p, p] = cos
        R[q, q] = cos
        R[p, q] = sin
        R[q, p] = -sin
        A = R.T.dot(A).dot(R)
        V = V.dot(R)
    eigenvalues = np.diag(A)
    return eigenvalues, V

def givens_rotation(A):
    """QR decomposition using Givens rotations."""
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    for i in range(n):
        for j in range(i+1, m):
            if R[j, i] != 0:
                r = np.hypot(R[i, i], R[j, i])
                cos = R[i, i] / r
                sin = -R[j, i] / r
                G = np.eye(m)
                G[[i, j], [i, j]] = cos
                G[i, j] = -sin
                G[j, i] = sin
                R = G.dot(R)
                Q = Q.dot(G.T)
    return Q, R

def householder_qr(A):
    """QR decomposition using Householder reflections."""
    m, n = A.shape
    R = A.copy().astype(float)
    Q = np.eye(m)
    for i in range(n):
        x = R[i:, i]
        normx = np.linalg.norm(x)
        if normx == 0:
            continue
        e1 = np.zeros_like(x)
        e1[0] = 1
        v = x + np.sign(x[0]) * normx * e1
        v = v / np.linalg.norm(v)
        H = np.eye(m)
        H[i:, i:] -= 2.0 * np.outer(v, v)
        R = H.dot(R)
        Q = Q.dot(H.T)
    return Q, R

def householder_tridiagonal(A):
    """Reduce symmetric matrix A to tridiagonal form using Householder reflections."""
    A = A.copy().astype(float)
    n = A.shape[0]
    for k in range(n-2):
        x = A[k+1:, k]
        normx = np.linalg.norm(x)
        if normx == 0:
            continue
        e1 = np.zeros_like(x)
        e1[0] = 1
        v = x - normx * e1
        v = v / np.linalg.norm(v)
        H = np.eye(n)
        H[k+1:, k+1:] -= 2.0 * np.outer(v, v)
        A = H.dot(A).dot(H)
    return A