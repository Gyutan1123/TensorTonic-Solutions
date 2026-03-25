import numpy as np

def get_alpha_bar(betas: np.ndarray) -> np.ndarray:
    """
    Compute cumulative product of (1 - beta).
    """
    return np.cumprod(1-betas)

def forward_diffusion(
    x_0: np.ndarray,
    t: int,
    betas: np.ndarray
) -> tuple:
    """
    Sample x_t from q(x_t | x_0).
    """

    alpha_bar_t = get_alpha_bar(betas)[t]
    shape = (len(t),) + (1,) * (x_0.ndim-1)
    alpha_bar_t = alpha_bar_t.reshape(shape)
    e = np.random.normal(size=x_0.shape)
    x_t = np.sqrt(alpha_bar_t)*x_0 + np.sqrt(1-alpha_bar_t)*e

    return x_t, e

def compute_ddpm_loss(
    model_predict: callable,
    x_0: np.ndarray,
    betas: np.ndarray,
    T: int
) -> float:
    """
    Compute DDPM training loss for a batch of images.
    """
    t = np.random.randint(low=1, high=(T+1), size=x_0.shape[0])

    x_t, eps = forward_diffusion(x_0, t, betas)

    eps_pred = model_predict(x_t, t)

    loss = np.mean((eps-eps_pred) ** 2)

    return loss
    