#!/bin/bash
################################################################################
# RUN GRIDWORLD DQN EXPERIMENTS
################################################################################
mkdir results/GRIDWORLD
# BASELINE SIMULATIONS
python run_learning_grid.py --RUN_TIMES 5 --SAVE_FNAME base_stats.csv
python run_learning_grid.py --DOUBLE --RUN_TIMES 5 --SAVE_FNAME double_stats.csv
python run_learning_grid.py --AGENT MLP-Dueling-DQN --RUN_TIMES 5 --SAVE_FNAME stats.csv
python run_learning_grid.py --AGENT MLP-Dueling-DQN --DOUBLE --RUN_TIMES 5 --SAVE_FNAME double_stats.csv
# EXPERT GRAMMAR SIMULATIONS
python run_learning_grid.py --RUN_TIMES 5 --SAVE_FNAME base_stats.csv
# ONLINE GRAMMAR SIMULATIONS
