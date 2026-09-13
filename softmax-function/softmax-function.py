import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    # Write code here
    x = np.asarray(x)

    exp_values = np.exp(x - np.max(x, axis = -1, keepdims = True))

    return exp_values / np.sum(exp_values, axis = -1, keepdims = True)