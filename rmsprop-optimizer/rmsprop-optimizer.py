import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    s_new =  beta*np.array(s) + (1-beta)*np.square(g)
    w_new = w - (lr/(np.sqrt(np.array(s_new) + eps)))*g
    return w_new, s_new