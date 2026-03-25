import numpy as np

def linear_beta_schedule(T: int, beta_1: float = 0.0001, beta_T: float = 0.02) -> np.ndarray:
    """
    Linear noise schedule from beta_1 to beta_T.
    """
    return np.linspace(beta_1, beta_T, T)

def cosine_alpha_bar_schedule(T: int, s: float = 0.008) -> np.ndarray:
    """
    Cosine schedule for alpha_bar (cumulative signal retention).
    """

    def f(t):
        return (np.cos((t/T + s) / (1+s) * np.pi / 2)) **2

    t = np.arange(1,T+1)
    alpha_bar = f(t) / f(0)

    return alpha_bar
    

def alpha_bar_to_betas(alpha_bars: np.ndarray) -> np.ndarray:
    """
    Convert alpha_bar schedule to beta schedule.
    """
    alphas = np.concatenate(([alpha_bars[0]], alpha_bars[1:] / alpha_bars[:-1]))

    return 1 - alphas 
