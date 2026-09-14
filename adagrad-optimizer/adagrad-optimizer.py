import numpy as np

def adagrad_step(w: list, g: list, G: list, lr: float = 0.01, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w and new_G.
    """
    # Write code here
    G = np.asarray(G, dtype=float)
    g = np.asarray(g, dtype=float)
    w = np.asarray(w, dtype=float)
    

    G += g**2

    return {
        "new_w": w - lr*g/(np.sqrt(G + eps)),
        "new_G": G
    }