#!/bin/bash

python aul.py --pro_data='../dataset/tmp/lan1_out_shuffled_unmask.csv'\
              --anti_data='../dataset/tmp/reverse_lan1_out_shuffled.csv' \
              --output='../output/output.txt' \
              --model='mbert' \
              --method aula
