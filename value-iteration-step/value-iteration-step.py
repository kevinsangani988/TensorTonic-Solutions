def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """
    v_new = []
    for idx,state in enumerate(transitions):
        action_value = []
        for idx_st, action_prob in enumerate(state):
            value = 0
            for idx_act_prob, prob in enumerate(action_prob):
                value += prob*(rewards[idx][idx_st] +gamma*values[idx_act_prob])
            action_value.append(value)
        v_new.append(max(action_value))
    return v_new