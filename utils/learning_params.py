# Define all required learning hyperparameters

def learning_parameters(l_type):
    if l_type == "Q-Learning":
        params = {"alpha": 0.8,  # Learning rate
                  "gamma": 0.95,  # Discount factor
                  "lambd": 0.1,  # TD(lambda) exponential decay factor
                  "epsilon": 0.1}  # Exploration parameter

    elif l_type == "SMDP-Q-Learning":
        params = {"alpha": 0.8,  # Learning rate
                  "gamma": 0.95,  # Discount factor
                  "lambd": 0.1,  # TD(lambda) exponential decay factor
                  "epsilon": 0.1}  # Exploration parameter

    return params
