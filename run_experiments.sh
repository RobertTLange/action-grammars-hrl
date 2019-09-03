#!/bin/bash
################################################################################
# SETUP OF EXPERIMENTS
################################################################################
if [[ "$*" == "setup" ]]
then
    echo "Run Installation Setup"
    # Clone & install the Towers of Hanoi Environment
    git clone https://github.com/RobertTLange/gym-hanoi
    cd gym_hanoi
    python setup.py install
    # Create empty results folder
    mkdir -p results/TOH
    mkdir -p results/GRIDWORLD
    mkdir -p results/ATARI
    # Setup the Grammar induction pipeline
    cd ..
    cd grammars
    # Clone Sequitur and Install - Forked & restructured from craignm @29/07/19
    git clone https://github.com/RobertTLange/sequitur
    cd sequitur
    make
################################################################################
# RUN TOWERS OF HANOI EXPERIMENTS - SAVES AUTOMATICALLY TO CSV FILES
################################################################################
elif [[ "$*" == "toh-fig6-left" ]]
then
    echo "Run Towers of Hanoi Tabular Experiments (figure 6 left plot)"
    # 5 DISK ENVIRONMENT (TD baseline, Imitation, Transfer from 4 disk)
    python run_learning_towers.py --N_DISKS 5 --LEARN_TYPE Q-Learning --RUN_TIMES 5
    python run_learning_towers.py --N_DISKS 5 --LEARN_TYPE Imitation-SMDP-Q-Learning --RUN_TIMES 5
    python run_learning_towers.py --N_DISKS 5 --LEARN_TYPE Transfer-SMDP-Q-Learning --RUN_TIMES 5
elif [[ "$*" == "toh-fig6-middle" ]]
then
    echo "Run Towers of Hanoi Tabular Experiments (figure 6 middle plot)"
    # 6 DISK ENVIRONMENT (TD baseline, Imitation, Transfer from 5 & 4 disk)
    python run_learning_towers.py --N_DISKS 6 --LEARN_TYPE Q-Learning --RUN_TIMES 5
    python run_learning_towers.py --N_DISKS 6 --LEARN_TYPE Imitation-SMDP-Q-Learning  --RUN_TIMES 5
    python run_learning_towers.py --N_DISKS 6 --LEARN_TYPE Transfer-SMDP-Q-Learning --TRANSFER_DISTANCE 1  --RUN_TIMES 5
    python run_learning_towers.py --N_DISKS 6 --LEARN_TYPE Transfer-SMDP-Q-Learning --TRANSFER_DISTANCE 2  --RUN_TIMES 5
elif [[ "$*" == "toh-fig6-right" ]]
then
    echo "Run Towers of Hanoi Tabular Experiments (figure 6 right plot)"
    # TODO: 6 DISK ENVIRONMENT 3 Sequitur and G-Lexis
    python run_learning_towers.py --N_DISKS 6 --LEARN_TYPE Imitation-SMDP-Q-Learning  --RUN_TIMES 5 --GRAMMAR_TYPE 3-Sequitur --SAVE_FNAME 3_seq_TOH.csv
    python run_learning_towers.py --N_DISKS 6 --LEARN_TYPE Imitation-SMDP-Q-Learning  --RUN_TIMES 5 --GRAMMAR_TYPE G-Lexis --SAVE_FNAME g_lexis_TOH.csv
elif [[ "$*" == "toh-online" ]]
then
    # TODO: ONLINE GRAMMAR EXPERIMENTS 5+6 Disk Environment Schedule k?!
    echo "Run ToH Tabular Online Grammar Experiments (figure 8 r1 left+middle)"
################################################################################
# RUN GRIDWORLD DQN EXPERIMENTS
################################################################################
elif [[ "$*" == "grid-experts" ]]
then
    echo "Train Grid Experts 250k, 500k, 1M"
    python run_learning_grid.py --RUN_TIMES 1 --SAVE --VERBOSE
elif [[ "$*" == "grid-fig7-left" ]]
then
    echo "Run Grid Expert Grammar-DQN Experiments (fig 7 left)"
    # BASELINE SIMULATION (DQN + Dueling DQN)
    python run_learning_grid.py --RUN_TIMES 5 --SAVE_FNAME dqn_stats.csv
    python run_learning_grid.py --AGENT MLP-Dueling-DQN --RUN_TIMES 5 --SAVE_FNAME dueling_stats.csv
    # EXPERT GRAMMAR SIMULATIONS (DQN AGENT AFTER 1 MIO ITS)
    python run_learning_grid.py --RUN_TIMES 5 --RUN_EXPERT_GRAMMAR --LOAD_CKPT agents/trained/1000000_mlp_agent.pt --SAVE_FNAME grid_expert_1M.csv
elif [[ "$*" == "grid-fig7-middle" ]]
then
    echo "Run Grid Transfer Grammar-DQN Experiments (fig 7 middle)"
    # TRANSFER GRAMMAR SIMULATIONS (DQN AGENT AFTER 250/500K ITS)
    python run_learning_grid.py --RUN_TIMES 5 --RUN_EXPERT_GRAMMAR --LOAD_CKPT agents/trained/250000_mlp_agent.pt --SAVE_FNAME grid_transfer_250k.csv
    python run_learning_grid.py --RUN_TIMES 5 --RUN_EXPERT_GRAMMAR --LOAD_CKPT agents/trained/500000_mlp_agent.pt --SAVE_FNAME grid_transfer_500k.csv
elif [[ "$*" == "gridworld-online" ]]
then
    echo "Run Gridworld Online Grammar-DQN Experiments (fig 8 r1 right)"
    # ONLINE GRAMMAR SIMULATIONS
    python run_learning_grid.py --AGENT MLP-Dueling-DQN --RUN_TIMES 5 --SAVE_FNAME grid_online_dueling.csv --RUN_ONLINE_GRAMMAR
################################################################################
# RUN ATARI DQN EXPERIMENTS
################################################################################
elif [[ "$*" == "pong-atari-expert" ]]
then
    echo "Run Pong ATARI Expert Grammar-DQN Experiments (fig 7 right)"
    # BASELINE SIMULATIONS
    python run_learning_atari.py --RUN_TIMES 3 --SAVE_FNAME base_stats.csv
elif [[ "$*" == "pong-atari-online" ]]
then
    echo "Run Pong ATARI Online Grammar-DQN Experiments (fig 8 r2 left)"
    # BASELINE SIMULATIONS
    # TODO: Online Pong
elif [[ "$*" == "seaquest-atari-online" ]]
then
    echo "Run Seaquest ATARI Online Grammar-DQN Experiments (fig 8 r2 middle)"
    # BASELINE SIMULATIONS
    # TODO: Online SeaQuest
elif [[ "$*" == "mspacman-atari-online" ]]
then
    echo "Run Ms Pacman ATARI Online Grammar-DQN Experiments (fig 8 r2 right)"
    # BASELINE SIMULATIONS
    # TODO: Online MsPacman
fi
