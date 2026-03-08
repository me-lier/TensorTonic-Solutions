import numpy as np

def fun(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here

    ans = []

    if isinstance(x, list):
        for i in x:
            if isinstance(x, list):
                ans.append(tanh(i))
            else:
                ans.append(fun(i))

    return ans if ans != [] else fun(x)
