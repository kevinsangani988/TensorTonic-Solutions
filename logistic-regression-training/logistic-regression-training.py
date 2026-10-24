import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    m, d = X.shape
    w = np.zeros(d)
    b = 0

    for i in range(steps):
        X_pred = np.dot(X,w) + b
        X_pred = _sigmoid(X_pred)

        dw = (1/m)*np.dot(X.T, (X_pred-y))
        db = (1/m)*np.sum(X_pred-y)

        w = w - lr*dw
        b = b - lr*db

    
    return w,b