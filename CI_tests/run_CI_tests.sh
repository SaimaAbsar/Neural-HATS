#!/bin/bash

pip3 install -r ./../requirements.txt 
python main_CI_1.py --data './../Data/6_ER_new0.01.txt' --graph './../Data/A_6_ER_new.npy' --output './outputs/CI_table_ER1.txt'

#python main_CI_0.py --data './../Data/Netsim/sim5/sim5_1.txt' --graph '/Data/Netsim/sim5/gt_sim5.npy' --output '/outputs/CI_0_sim5_1.txt'