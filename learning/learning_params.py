# Define all required learning hyperparameters


def learning_parameters(l_type, NUM_DISKS=4):
    if l_type == "Q-Learning":
        params = {"ALPHA": 0.8,  # Learning rate
                  "GAMMA": 0.95,  # Discount factor
                  "LAMBDA": 0.1,  # TD(lambda) exponential decay factor
                  "EPSILON": 0.1}  # Exploration parameter

    elif l_type == "Imitation-SMDP-Q-Learning":
        params = {"ALPHA": 0.8,  # Learning rate
                  "GAMMA": 0.95,  # Discount factor
                  "LAMBDA": 0.,  # TD(lambda) exponential decay factor
                  "EPSILON": 0.1}  # Exploration parameter

    elif l_type == "Transfer-SMDP-Q-Learning":
        params = {"ALPHA": 0.8,  # Learning rate
                  "GAMMA": 0.95,  # Discount factor
                  "LAMBDA": 0.,  # TD(lambda) exponential decay factor
                  "EPSILON": 0.1}  # Exploration parameter

    elif l_type == "Online-SMDP-Q-Learning":
        params = {"ALPHA": 0.8,  # Learning rate
                  "GAMMA": 0.95,  # Discount factor
                  "LAMBDA": 0.,  # TD(lambda) exponential decay factor
                  "EPSILON": 0.1 # Exploration parameter
                  }

        if num_disks == 4:
            params[ "init_q_eps"] = 10
            params["inter_update_eps"] =  25
            params["num_grammar_updates"] = 12

    if NUM_DISKS == 4:
        params["NUM_UPDATES"] = 11000
        params["MAX_STEPS"] = 500
    elif NUM_DISKS == 5:
        params["NUM_UPDATES"] = 17000
        params["MAX_STEPS"] = 2000
    elif NUM_DISKS == 6:
        params["NUM_UPDATES"] = 600000
        params["MAX_STEPS"] = 5000

    return params
