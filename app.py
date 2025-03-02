from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from utils.graph_generator import generate_graphical_plot
from utils.numerical_methods import (
    bisection_method,
    newton_raphson_method,
    secant_method,
    fixed_point_iteration,
    false_position_method,
    muller_method,
    jacobi_method,
    iterative_matrix_inverse
)
from utils.approximations_and_errors import calculate_absolute_error, calculate_relative_error
from utils.interpolation import newton_forward_interpolation
from utils.numerical_integrations import trapezoidal_rule
import sympy as sp
import numpy as np
import matplotlib as plt

app = Flask(__name__)
CORS(app)

def evaluate_function(function_str):
    """
    Converts a string like "x**2 + 3*x - 5" into a callable Python function f(x).
    """
    x = sp.Symbol('x')
    expr = sp.sympify(function_str)
    return sp.lambdify(x, expr, "numpy")

# --------------------- Plot Graph (Task 1) ---------------------
@app.route("/plot_graph", methods=["POST"])
def plot_graph():
    data = request.json
    func_str = data.get("function")
    range_str = data.get("range_x")
    if not func_str or not range_str:
        return jsonify({"error": "Missing function or range_x"}), 400

    f = evaluate_function(func_str)
    x_range = range_str.split()
    return generate_graphical_plot(f, func_str, x_range)

# --------------------- Bisection Method ---------------------
@app.route("/calculate-bisection", methods=["POST"])
def calculate_bisection():
    data = request.json
    try:
        func_str = data["function"]
        f = evaluate_function(func_str)
        a = float(data["initial_guess_1"])
        b_val = float(data["initial_guess_2"])
        tol = float(data["tolerance"])
        root = bisection_method(f, a, b_val, tol)
        return jsonify({"true_root": root})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------- Newton-Raphson Method ---------------------
@app.route("/calculate-newton-raphson", methods=["POST"])
def calculate_newton_raphson():
    data = request.json
    try:
        func_str = data["function"]
        derivative_str = data["derivative"]
        x0 = float(data["x0"])
        tol = float(data["tolerance"])

        f = evaluate_function(func_str)
        df = evaluate_function(derivative_str)

        root = newton_raphson_method(f, df, x0, tol)
        return jsonify({"root": root})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------- Secant Method ---------------------
@app.route("/calculate-secant", methods=["POST"])
def calculate_secant():
    data = request.json
    try:
        func_str = data["function"]
        x0 = float(data["initial_guess_1"])
        x1 = float(data["initial_guess_2"])
        tol = float(data["tolerance"])

        f = evaluate_function(func_str)
        root = secant_method(f, x0, x1, tol)
        return jsonify({"root": root})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------- Fixed Point Iteration ---------------------
@app.route("/calculate-fixed-point", methods=["POST"])
def calculate_fixed_point():
    data = request.json
    try:
        g_str = data["g"]  # user provides g(x) in 'function' field
        x0 = float(data["x0"])
        tol = float(data["tolerance"])

        g = evaluate_function(g_str)
        root = fixed_point_iteration(g, x0, tol)
        return jsonify({"root": root})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------- False Position Method ---------------------
@app.route("/calculate-false-position", methods=["POST"])
def calculate_false_position():
    data = request.json
    try:
        func_str = data["function"]
        a = float(data["a"])
        b = float(data["b"])
        tol = float(data["tolerance"])

        f = evaluate_function(func_str)
        root = false_position_method(f, a, b, tol)
        return jsonify({"root": root})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------- Muller's Method ---------------------
@app.route("/calculate-muller", methods=["POST"])
def calculate_muller():
    data = request.json
    try:
        func_str = data["function"]
        x0 = float(data["x0"])
        x1 = float(data["x1"])
        x2 = float(data["x2"])
        tol = float(data["tolerance"])

        f = evaluate_function(func_str)
        root = muller_method(f, x0, x1, x2, tol)
        root_str = str(root)
        return jsonify({"root": root_str})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------- Absolute Error Calculation ---------------------
@app.route("/calculate-absolute-error", methods=["POST"])
def calculate_abs_error():
    data = request.json
    try:
        true_root = float(data["true_root"])
        approx_root = float(data["approximate_root"])
        abs_error = calculate_absolute_error(true_root, approx_root)
        return jsonify({"absolute_error": abs_error})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------- Task 2: Comparison of Root-Finding Methods ---------------------

@app.route("/compare-root-finding", methods=["POST"])
def compare_root_finding():
    data = request.json
    try:
        f = evaluate_function(data.get("function"))
        a = float(data.get("initial_guess_1"))
        b_val = float(data.get("initial_guess_2"))
        x0_secant = a  # for secant method, you may choose appropriate values
        x1_secant = b_val
        tol = float(data.get("tolerance"))
        # Calculate roots and assume that your functions return iteration count if needed
        root_bisect, iter_bisect = bisection_method(f, a, b_val, tol, return_iterations=True)
        root_secant, iter_secant = secant_method(f, x0_secant, x1_secant, tol, return_iterations=True)
        rel_error_bisect = calculate_relative_error(root_bisect, float(data.get("approximate_root", root_bisect)))
        rel_error_secant = calculate_relative_error(root_secant, float(data.get("approximate_root", root_secant)))
        return jsonify({
            "bisection": {"root": root_bisect, "iterations": iter_bisect, "relative_error": rel_error_bisect},
            "secant": {"root": root_secant, "iterations": iter_secant, "relative_error": rel_error_secant}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------- Task 3: Jacobi Method ---------------------
@app.route("/solve-jacobi",methods=["POST"])
def solve_jacobi():
    data = request.json
    try:
        # Expecting matrix A and vector b in JSON (as lists)
        A = np.array(data.get("A"))
        b = np.array(data.get("b"))
        x0 = np.array(data.get("x0"))
        tol = float(data.get("tol", 1e-6))
        max_iter = int(data.get("max_iter", 100))
        solution = jacobi_method(A, b, x0, tol, max_iter)
        return jsonify({"solution": solution.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------- Task 4: Iterative Matrix Inversion ---------------------
@app.route("/iterative-matrix-inverse", methods=["POST"])
def inverse_matrix():
    data = request.json
    try:
        A = np.array(data.get("A"))
        tol = float(data.get("tol", 1e-6))
        max_iter = int(data.get("max_iter", 100))
        # Use your iterative_matrix_inverse function from numerical_methods.py
        A_inv = iterative_matrix_inverse(A, tol, max_iter)
        I = np.dot(A, A_inv)
        return jsonify({"inverse": A_inv.tolist(), "identity_matrix": I.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------- Task 5: Linear Curve Fitting ---------------------
@app.route("/linear-curve-fit", methods=["POST"])
def curve_fit():
    data = request.json
    try:
        # Expecting data points x and y as lists
        x = np.array(data.get("x"))
        y = np.array(data.get("y"))
        # Use least squares (np.polyfit) for linear fit (degree=1)
        coefficients = np.polyfit(x, y, 1)
        m, c = coefficients
        # Optionally, generate fitted line points for plotting
        fitted_y = m * x + c
        
        plt.scatter(x, y, color='red', label='Data Points')  # Scatter plot for data
        plt.plot(x, fitted_y, label=f'Best Fit: y = {m:.2f}x + {c:.2f}', color='blue')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.title('Linear Curve Fitting')
        plt.grid()
        


        return jsonify({"slope": m, "intercept": c, "fitted_y": fitted_y.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------- Task 6: Newton’s Forward Interpolation ---------------------
@app.route("/newton-forward-interpolation", methods=["POST"])
def newton_forward():
    data = request.json
    try:
        # Expecting x, y data points and a value to interpolate
        x = np.array(data.get("x"))
        y = np.array(data.get("y"))
        x_value = float(data.get("x_value"))
        interpolated_value = newton_forward_interpolation(x, y, x_value)
        return jsonify({"interpolated_value": interpolated_value})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------- Task 7: First Derivative Using Forward Difference ---------------------
@app.route("/first-derivative", methods=["POST"])
def first_derivative():
    data = request.json
    try:
        x = np.array(data.get("x"))
        y = np.array(data.get("y"))
        x_value = float(data.get("x_value"))
        derivative = newton_forward_interpolation(x, y, x_value)
        return jsonify({"derivative": derivative})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------- Task 8: Trapezoidal Rule for Integration ---------------------
@app.route("/trapezoidal-rule", methods=["POST"])
def integrate_trapezoidal():
    data = request.json
    try:
        func_str = data.get("function")
        f = evaluate_function(func_str)
        a, b = map(float, data.get("interval").split())
        n = int(data.get("subintervals"))
        integral_value = trapezoidal_rule(f, a, b, n)
        return jsonify({"integral": integral_value})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)