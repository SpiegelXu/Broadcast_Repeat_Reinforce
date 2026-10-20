import numpy as np

def generate_cos_poly(
    total_points=150,
    history_len=100,
    degree=3,
    coeffs=None,
    normalize_x=True,
    x_range=(0, 150),
    noise=0.0,
    seed=None
):
    
    rng = np.random.RandomState(seed)

    x_full = np.linspace(x_range[0], x_range[1], total_points)

    if coeffs is None:
        coeffs_used = [rng.uniform(-1e-3, 1e-3) for _ in range(degree + 1)]
    else:
        coeffs_used = coeffs

    if normalize_x:
        x_mapped = (x_full - x_range[0]) / (x_range[1] - x_range[0]) * 2 * np.pi
    else:
        x_mapped = x_full

    poly_val = np.zeros_like(x_mapped)
    for i, c in enumerate(coeffs_used):
        poly_val += c * (x_mapped ** i)

    y_clean = np.cos(poly_val)

    y_full = y_clean + rng.normal(0, noise, size=total_points)

    x_hist, y_hist = x_full[:history_len], y_full[:history_len]
    x_pred, y_pred = x_full[history_len:], y_full[history_len:]

    return x_full, y_full, x_hist, y_hist, x_pred, y_pred, coeffs_used
