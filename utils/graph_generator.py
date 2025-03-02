# graph_generator.py

import matplotlib.pyplot as plt
import numpy as np
import io, base64
from flask import jsonify

def generate_graphical_plot(f, func_str, x_range):
    """
    Generate a plot for a function f(x) over a given range.
    
    Parameters:
        f: a callable function (e.g. from sympy.lambdify)
        func_str: string representation of the function (for labeling)
        x_range: a list or tuple of two values [x_min, x_max]
    
    Returns:
        A Flask JSON response containing the plot image (base64 encoded) or an error.
    """
    try:
        x_min, x_max = map(float, x_range)
        x = np.linspace(x_min, x_max, 500)
        y = f(x)
        plt.figure(figsize=(5, 6))
        plt.plot(x, y, label=f"f(x) = {func_str}")
        plt.axhline(0, color='red', linestyle='--', label="y = 0")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Graphical Method")
        plt.legend()
        plt.grid(True)
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return jsonify({"image": base64.b64encode(img.getvalue()).decode()})
    except Exception as e:
        return jsonify({"error": str(e)})