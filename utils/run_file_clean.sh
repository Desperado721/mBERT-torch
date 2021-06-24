python file_clean_equal.py -lan0_s_in '/data/zijian/Bang/Gebiotoolkit_ori/corpus_alignment/results_new/lan_0_she.txt' \
                     -lan1_s_in '/data/zijian/Bang/Gebiotoolkit_ori/corpus_alignment/results_new/lan_1_she.txt' \
                     -lan0_h_in '/data/zijian/Bang/Gebiotoolkit_ori/corpus_alignment/results_new/lan_0_he.txt' \
                     -lan1_h_in '/data/zijian/Bang/Gebiotoolkit_ori/corpus_alignment/results_new/lan_1_he.txt' \
                     -lan0_h_out '/data/zijian/Bang/mBERT-torch/dataset/tmp/reverse_lan0_he_out.csv' \
                     -lan1_h_out '/data/zijian/Bang/mBERT-torch/dataset/tmp/reverse_lan1_he_out.csv' \
                     -lan0_s_out '/data/zijian/Bang/mBERT-torch/dataset/tmp/reverse_lan0_she_out.csv' \
                     -lan1_s_out '/data/zijian/Bang/mBERT-torch/dataset/tmp/reverse_lan1_she_out.csv' \
                     -shuffle \
                     -shuffled_lan0_out '/data/zijian/Bang/mBERT-torch/dataset/tmp/reverse_lan0_out_shuffled.csv' \
                     -shuffled_lan1_out '/data/zijian/Bang/mBERT-torch/dataset/tmp/reverse_lan1_out_shuffled.csv' \
                     -shuffled_lan0_out_ori '/data/zijian/Bang/mBERT-torch/dataset/tmp/lan0_out_shuffled_unmask.csv' \
                     -shuffled_lan1_out_ori '/data/zijian/Bang/mBERT-torch/dataset/tmp/lan1_out_shuffled_unmask.csv' \
                     