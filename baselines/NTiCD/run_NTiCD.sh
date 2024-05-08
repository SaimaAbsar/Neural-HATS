#!/bin/bash

pip3 install -r ./../../requirements.txt 
python main.py --datapath './../../Data/6_ER_new0.01.txt' --d 6 --graph './../../Data/A_6_ER_new.npy' --ci './../../CI_tests/outputs/CI_table_ER1.txt'