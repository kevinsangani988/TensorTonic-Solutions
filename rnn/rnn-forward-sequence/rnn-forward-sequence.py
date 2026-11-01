import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray, W_xh: np.ndarray,
                W_hh: np.ndarray, b_h: np.ndarray) -> dict:
    """
    Returns hidden_states and final_hidden_state as float64 arrays.
    """
    hidden = h_0.copy()
    state = []
    for states in range(X.shape[1]):
        hidden = np.tanh(X[:,states,:]@W_xh.T + hidden@W_hh.T + b_h )
        state.append(hidden)
    return {"hidden_states": np.stack(state,axis=1).astype(np.float64), "final_hidden_state": hidden.astype(np.float64)}