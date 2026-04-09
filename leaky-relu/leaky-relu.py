import numpy as np


def leaky_relu(x, alpha=0.01):
    ans=[alpha*i if i<0 else i for i in x ]
    return np.array(ans)