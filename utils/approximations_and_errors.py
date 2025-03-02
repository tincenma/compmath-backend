# approximations_and_errors.py

def calculate_absolute_error(true_value, approx_value):
    """Calculate absolute error: |true_value - approx_value|."""
    return abs(true_value - approx_value)

def calculate_relative_error(true_value, approx_value):
    """Calculate relative error: |true_value - approx_value| / |true_value|."""
    if true_value == 0:
        return float('inf')
    return abs(true_value - approx_value) / abs(true_value)

def calculate_percentage_error(true_value, approx_value):
    """Calculate percentage error."""
    return calculate_relative_error(true_value, approx_value) * 100

def count_significant_digits(number):
    """Count the number of significant digits in a number."""
    num_str = str(number).lstrip('0').replace('.', '')
    return len(num_str.rstrip('0'))

def round_to_significant_figures(number, n):
    """Round a number to n significant figures."""
    if number == 0:
        return 0
    else:
        scale = -int(f"{number:e}".split('e')[1]) + (n - 1)
        return round(number, scale)

def error_propagation_addition(a, b, error_a, error_b):
    """Calculate propagated error for addition: result ± (error_a + error_b)."""
    result = a + b
    propagated_error = error_a + error_b
    return result, propagated_error

def error_propagation_multiplication(a, b, error_a, error_b):
    """Calculate propagated error for multiplication using absolute errors."""
    result = a * b
    propagated_error = abs(b) * error_a + abs(a) * error_b
    return result, propagated_error

def truncate_number(number, decimal_places):
    """Truncate a number to a specified number of decimal places (without rounding)."""
    multiplier = 10 ** decimal_places
    return int(number * multiplier) / multiplier

def compare_rounding_and_truncation(number, decimal_places):
    """
    Return a tuple of (truncated value, rounded value) for the given number and decimal places.
    """
    truncated = truncate_number(number, decimal_places)
    rounded = round(number, decimal_places)
    return truncated, rounded