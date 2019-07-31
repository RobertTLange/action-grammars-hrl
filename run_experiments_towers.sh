#!/bin/bash
################################################################################
# RUN TOWERS OF HANOI EXPERIMENTS - SAVES AUTOMATICALLY TO CSV FILES
################################################################################
mkdir results/TOH
# 5 DISK ENVIRONMENT
python run_learning_towers.py --N_DISKS 5 --LEARN_TYPE Q-Learning --RUN_TIMES 5
python run_learning_towers.py --N_DISKS 5 --LEARN_TYPE Imitation-SMDP-Q-Learning --RUN_TIMES 5
python run_learning_towers.py --N_DISKS 5 --LEARN_TYPE Transfer-SMDP-Q-Learning --RUN_TIMES 5
# 6 DISK ENVIRONMENT
python run_learning_towers.py --N_DISKS 6 --LEARN_TYPE Q-Learning --RUN_TIMES 5
python run_learning_towers.py --N_DISKS 6 --LEARN_TYPE Imitation-SMDP-Q-Learning  --RUN_TIMES 5
python run_learning_towers.py --N_DISKS 6 --LEARN_TYPE Transfer-SMDP-Q-Learning --TRANSFER_DISTANCE 1  --RUN_TIMES 5
python run_learning_towers.py --N_DISKS 6 --LEARN_TYPE Transfer-SMDP-Q-Learning --TRANSFER_DISTANCE 2  --RUN_TIMES 5
