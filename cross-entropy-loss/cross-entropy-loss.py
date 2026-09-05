import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    # Write code here
    y_pred = np.asarray(y_pred)
    # y_pred = np.dot(y_true, y_pred)
    # print(y_pred)


    return np.mean(-np.log(y_pred[
        range(len(y_pred)), y_true
    ]))