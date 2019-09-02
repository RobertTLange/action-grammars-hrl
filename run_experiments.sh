#!/bin/bash
if [[ "$*" == "setup" ]]
then
    ################################################################################
    # SETUP OF EXPERIMENTS
    ################################################################################
    # Clone & install the Towers of Hanoi Environment
    git clone https://github.com/RobertTLange/gym-hanoi
    cd gym_hanoi
    python setup.py install
    # Create empty results folder
    mkdir results
    # Setup the Grammar induction pipeline
    cd ..
    cd grammars
    # Clone Sequitur and Install - Forked from craignm @29/07/19
    git clone https://github.com/RobertTLange/sequitur
    cd sequitur
    make
elif [[ "$*" == "towers-of-hanoi" ]]
then
    ################################################################################
    # RUN TOWERS OF HANOI EXPERIMENTS - SAVES AUTOMATICALLY TO CSV FILES
    ################################################################################
    echo "Run Towers of Hanoi Tabular Experiments"
    mkdir -p results/TOH
    # 5 DISK ENVIRONMENT
    python run_learning_towers.py --N_DISKS 5 --LEARN_TYPE Q-Learning --RUN_TIMES 5
    python run_learning_towers.py --N_DISKS 5 --LEARN_TYPE Imitation-SMDP-Q-Learning --RUN_TIMES 5
    python run_learning_towers.py --N_DISKS 5 --LEARN_TYPE Transfer-SMDP-Q-Learning --RUN_TIMES 5
    # 6 DISK ENVIRONMENT
    python run_learning_towers.py --N_DISKS 6 --LEARN_TYPE Q-Learning --RUN_TIMES 5
    python run_learning_towers.py --N_DISKS 6 --LEARN_TYPE Imitation-SMDP-Q-Learning  --RUN_TIMES 5
    python run_learning_towers.py --N_DISKS 6 --LEARN_TYPE Transfer-SMDP-Q-Learning --TRANSFER_DISTANCE 1  --RUN_TIMES 5
    python run_learning_towers.py --N_DISKS 6 --LEARN_TYPE Transfer-SMDP-Q-Learning --TRANSFER_DISTANCE 2  --RUN_TIMES 5
    # ONLINE EXPERIMENTS
elif [[ "$*" == "gridworld" ]]
then
    ################################################################################
    # RUN GRIDWORLD DQN EXPERIMENTS
    ################################################################################
    echo "Run Gridworld Grammar-DQN Experiments"
    mkdir -p results/GRIDWORLD
    # BASELINE SIMULATIONS
    python run_learning_grid.py --RUN_TIMES 5 --SAVE_FNAME base_stats.csv
    python run_learning_grid.py --DOUBLE --RUN_TIMES 5 --SAVE_FNAME double_stats.csv
    python run_learning_grid.py --AGENT MLP-Dueling-DQN --RUN_TIMES 5 --SAVE_FNAME stats.csv
    python run_learning_grid.py --AGENT MLP-Dueling-DQN --DOUBLE --RUN_TIMES 5 --SAVE_FNAME double_stats.csv
    # EXPERT GRAMMAR SIMULATIONS
    python run_learning_grid.py --RUN_TIMES 5 --SAVE_FNAME base_stats.csv
    # ONLINE GRAMMAR SIMULATIONS
elif [[ "$*" == "atari" ]]
then
    ################################################################################
    # RUN GRIDWORLD DQN EXPERIMENTS
    ################################################################################
    echo "Run ATARI Grammar-DQN Experiments"
    mkdir -p results/ATARI
    # BASELINE SIMULATIONS
    python run_learning_atari.py --RUN_TIMES 3 --SAVE_FNAME base_stats.csv
fi
