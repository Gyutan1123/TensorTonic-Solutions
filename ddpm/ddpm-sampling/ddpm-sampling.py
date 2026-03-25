import numpy as np

def get_alpha_bar(betas: np.ndarray) -> np.ndarray:
    """
    Compute cumulative product of (1 - beta).
    """
    return np.cumprod(1 - betas)

def reverse_step(
    x_t: np.ndarray,
    t: int,
    epsilon_pred: np.ndarray,
    betas: np.ndarray
) -> np.ndarray:
    """
    Perform one reverse diffusion step.
    """
    alpha_t = (1-betas)[t]
    alpha_bar_t = get_alpha_bar(betas)[t]
    u = (x_t - (1 - alpha_t) * epsilon_pred / np.sqrt(1-alpha_bar_t)) / np.sqrt(alpha_t)

    if t > 1:
        z = np.sqrt(betas[t]) * np.random.normal(size=x_t.shape)
    else:
        z = 0
    return u + z



def ddpm_sample(
    model_predict: callable,
    shape: tuple,
    betas: np.ndarray,
    T: int
) -> np.ndarray:
    """
    Generate a sample using DDPM.
    """
    x_t = np.random.normal(size=shape)

    for t in range(1, T+1, -1):
        eps_pred = model_predict(x_t, t)
        x_t1 = reverse_step(x_t, t, eps_pred, betas)

    return x_t
