import numpy as np

def fun(x):
    return 1/(1+np.exp(-x))

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    ans = []
    if isinstance(x, list):
        for i in x:
            if isinstance(i, list):
                ans.append(sigmoid(i))
            else:
                ans.append(fun(i))

    return ans if ans != [] else fun(x)
            
