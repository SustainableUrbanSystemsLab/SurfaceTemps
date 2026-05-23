import numpy as np


def h_convective(wind_speed: np.ndarray, model: str = "doe2") -> np.ndarray:
    """Exterior convective heat transfer coefficient (W/m2-K).

    DOE-2 linear model: h_c = 5.7 + 3.8 * v_wind
    """
    wind_speed = np.asarray(wind_speed, dtype=float)
    if model == "doe2":
        return 5.7 + 3.8 * wind_speed
    elif model == "constant":
        return np.full_like(wind_speed, 20.0)
    else:
        raise ValueError(f"Unknown convection model: {model}")
