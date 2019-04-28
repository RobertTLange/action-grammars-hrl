# Define all required learning hyperparameters


def learning_parameters(l_type, num_disks=4):
    if l_type == "Q-Learning":
        params = {"alpha": 0.8,  # Learning rate
                  "gamma": 0.95,  # Discount factor
                  "lambd": 0.1,  # TD(lambda) exponential decay factor
                  "epsilon": 0.1}  # Exploration parameter

    elif l_type == "Imitation-SMDP-Q-Learning":
        params = {"alpha": 0.8,  # Learning rate
                  "gamma": 0.95,  # Discount factor
                  "lambd": 0.,  # TD(lambda) exponential decay factor
                  "epsilon": 0.1}  # Exploration parameter

    elif l_type == "Transfer-SMDP-Q-Learning":
        params = {"alpha": 0.8,  # Learning rate
                  "gamma": 0.95,  # Discount factor
                  "lambd": 0.,  # TD(lambda) exponential decay factor
                  "epsilon": 0.1}  # Exploration parameter

    elif l_type == "Online-SMDP-Q-Learning":
        params = {"alpha": 0.8,  # Learning rate
                  "gamma": 0.95,  # Discount factor
                  "lambd": 0.,  # TD(lambda) exponential decay factor
                  "epsilon": 0.1 # Exploration parameter
                  }

        if num_disks == 4:
            params[ "init_q_eps"] = 10
            params["inter_update_eps"] =  25
            params["num_grammar_updates"] = 12

    else:
        raise ValueError("Provide valid learning type")

    return params
