# Sirivannavari : Princess Sirivannavari Nariratana Rajakanya ( ; ] ; ; born 8 January 1987) is a Princess of the Kingdom of Thailand and is the only daughter of King Vajiralongkorn and his former consort Sujarinee Vivacharawongse (commonly known as Yuvadhida Polpraserth) in the line of succession and to bear royal titles.
# Sirivannavari : A badminton tournament that made its debut in 2016, the Thailand Masters, was named as the Princess Sirivannavari Thailand Masters.
# Hu Xiaolian : She was ranked fourth on "The Wall Street Journal"' s "The 50 Women to Watch 2008" list and was referred to as "one of the most powerful people in the world".
# Hu Xiaolian : <templatestyles src="Reflist/styles.css" />
# Hu Xiaolian : Born in Hubei in 1958, Hu graduated from the Graduate School of the People's Bank of China with an MA in economics in 1984.

# 0 for chinese
# 1 for English

import argparse
import sys
import re
import os

def include_sentence(sens):
    name = sens.split(':')[0]
    valid_sentence = re.sub(name + ': ','',sens)
    valid_sentence = re.sub('\n','',valid_sentence)
    return valid_sentence, name[:len(name)-1]

def remove(path):
    if os.path.exists(path):
        os.remove(path)

def combine(args):
    with open(args.lan1_s_in) as f:
        she_sentences_lan1 = f.readlines()
    f.close()
    
    with open(args.lan1_h_in) as f:
        he_sentences_lan1 = f.readlines()
    f.close()

    with open(args.lan0_s_in) as f:
        she_sentences_lan0 = f.readlines()
    f.close()

    with open(args.lan0_h_in) as f:
        he_sentences_lan0 = f.readlines()
    f.close()
    
    all_she_sens1, all_she_sens0 = [],[]
    all_he_sens0, all_he_sens1 = [], []
    seen = set()

    remove(args.lan0_s_out)
    remove(args.lan1_s_out)
    remove(args.lan0_h_out)
    remove(args.lan1_h_out)

    assert len(she_sentences_lan0) == len(she_sentences_lan1) and len(he_sentences_lan0)==len(he_sentences_lan1), 'Please check the input to make sure they have same amount of sentences'
    she_amount = len(she_sentences_lan1)
    for j in range(she_amount):
        sens0,sens1 = she_sentences_lan0[j], she_sentences_lan1[j]
        valid_sentence1, current_name1 = include_sentence(she_sentences_lan1[j])
        valid_sentence0, current_name0 = include_sentence(she_sentences_lan0[j])
        # print(valid_sentence,current_name)

        logic1 = 'She' in valid_sentence1.split(' ') or 'she' in valid_sentence1.split(' ') or 'Her' in valid_sentence1.split(' ') or 'her' in valid_sentence1.split(' ')
        logic2 = '她' in valid_sentence0

        if logic1 or logic2:
            tmp1 = valid_sentence1.split(' ')

            tmp_out1 = ['<mask>' if s=='she' or s=='She' or s=='Her' or s=='her' else s for s in tmp1]
            all_she_sens1.append(' '.join(tmp_out1))
            seen.add(current_name1)

            valid_sentence0 = valid_sentence0.replace('她', '<性别>')
            all_she_sens0.append(valid_sentence0)
    
    with open(args.lan0_s_out, 'a') as f:
        for i in range(len(all_she_sens0)):
            f.write(all_she_sens0[i]+'0\n')
    with open(args.lan1_s_out, 'a') as f:
        for i in range(len(all_she_sens1)):
            f.write(all_she_sens1[i]+'0\n')
    print(len(all_she_sens0), len(all_she_sens1))
    # print('max_length',max([len(s.split(' ')) for s in all_she_sens]))
    
    # -------------------------he files----------------------------------------------------

    for j in range(len(he_sentences_lan0)):
        sens0,sens1 = he_sentences_lan1[j], he_sentences_lan1[j]
        valid_sentence0, current_name0 = include_sentence(he_sentences_lan0[j])
        valid_sentence1, current_name1 = include_sentence(he_sentences_lan1[j])

        logic0 = '他' in valid_sentence0
        logic1 = 'He' in valid_sentence1.split(' ') or 'he' in valid_sentence1.split(' ') or 'His' in valid_sentence1.split(' ') or 'his' in valid_sentence1.split(' ') or 'Him' in valid_sentence1.split(' ') or 'him' in valid_sentence1.split(' ')
        if logic0 or logic1:
            tmp = valid_sentence1.split(' ')

            tmp_out = ['<mask>' if s=='he' or s=='He' or s=='His' or s=='his' or s=='him' else s for s in tmp]
            valid_sentence0 = valid_sentence0.replace('他', '<性别>')
            all_he_sens1.append(' '.join(tmp_out))
            all_he_sens0.append(valid_sentence0)
            # seen.add(current_name0)  

    with open(args.lan0_h_out, 'a') as f:
        for i in range(len(all_he_sens0)):
            f.write(all_he_sens0[i]+'1\n')
    with open(args.lan1_h_out, 'a') as f:
        for i in range(len(all_he_sens1)):
            f.write(all_he_sens1[i]+'1\n')
    print(len(all_he_sens0), len(all_he_sens1))

    if args.shuffle:        
        with open(args.shuffled_lan0_out, 'a') as f:
            for i in range(len(all_she_sens0)):
                f.write(all_she_sens0[i]+'0\n')
                f.write(all_he_sens0[3*i]+'1\n')
                f.write(all_he_sens0[3*i+1]+'1\n')
                f.write(all_he_sens0[3*i+2]+'1\n')

        with open(args.shuffled_lan1_out, 'a') as f:
            for i in range(len(all_she_sens0)):
                f.write(all_she_sens1[i]+'0\n')
                f.write(all_he_sens1[3*i]+'1\n')
                f.write(all_he_sens1[3*i+1]+'1\n')
                f.write(all_he_sens1[3*i+2]+'1\n')


def combine_switch_gender(args):

    with open(args.lan1_s_in) as f:
        she_sentences_lan1 = f.readlines()
    f.close()
    
    with open(args.lan1_h_in) as f:
        he_sentences_lan1 = f.readlines()
    f.close()

    with open(args.lan0_s_in) as f:
        she_sentences_lan0 = f.readlines()
    f.close()

    with open(args.lan0_h_in) as f:
        he_sentences_lan0 = f.readlines()
    f.close()
    
    all_she_sens1, all_she_sens0 = [],[]
    all_he_sens0, all_he_sens1 = [], []
    seen = set()

    remove(args.lan0_s_out)
    remove(args.lan1_s_out)
    remove(args.lan0_h_out)
    remove(args.lan1_h_out)

    assert len(she_sentences_lan0) == len(she_sentences_lan1) and len(he_sentences_lan0)==len(he_sentences_lan1), 'Please check the input to make sure they have same amount of sentences'
    she_amount = len(she_sentences_lan1)
    for j in range(she_amount):
        sens0,sens1 = she_sentences_lan0[j], she_sentences_lan1[j]
        valid_sentence1, current_name1 = include_sentence(she_sentences_lan1[j])
        valid_sentence0, current_name0 = include_sentence(she_sentences_lan0[j])
        # print(valid_sentence,current_name)

        logic1 = 'She' in valid_sentence1.split(' ') or 'she' in valid_sentence1.split(' ') or 'Her' in valid_sentence1.split(' ') or 'her' in valid_sentence1.split(' ')
        logic2 = '她' in valid_sentence0
        she_sens_ori = []

        if logic1 or logic2:
            she_sens_ori.append(valid_sentence1)
            tmp1 = valid_sentence1.split(' ')

            # tmp_out1 = ['<mask>' if s=='she' or s=='She' or s=='Her' or s=='her' else s for s in tmp1]
            tmp_out1 = ['he' if s=='she' or s=='She' else s for s in tmp1]
            tmp_out2 = ['his' if s=='Her' or s=='her' else s for s in tmp_out1]

            all_she_sens1.append(' '.join(tmp_out2)) 
            seen.add(current_name1)

            valid_sentence0 = valid_sentence0.replace('她', '他')
            all_she_sens0.append(valid_sentence0)
    
    with open(args.lan0_s_out, 'a') as f:
        for i in range(len(all_she_sens0)):
            f.write(all_she_sens0[i]+'0\n')
    with open(args.lan1_s_out, 'a') as f:
        for i in range(len(all_she_sens1)):
            f.write(all_she_sens1[i]+'0\n')
    print(len(all_she_sens0), len(all_she_sens1))
    # print('max_length',max([len(s.split(' ')) for s in all_she_sens]))
    
    # -------------------------he files----------------------------------------------------

    for j in range(len(he_sentences_lan0)):
        sens0,sens1 = he_sentences_lan1[j], he_sentences_lan1[j]
        valid_sentence0, current_name0 = include_sentence(he_sentences_lan0[j])
        valid_sentence1, current_name1 = include_sentence(he_sentences_lan1[j])

        logic0 = '他' in valid_sentence0
        logic1 = 'He' in valid_sentence1.split(' ') or 'he' in valid_sentence1.split(' ') or 'His' in valid_sentence1.split(' ') or 'his' in valid_sentence1.split(' ') or 'Him' in valid_sentence1.split(' ') or 'him' in valid_sentence1.split(' ')
        he_sens_ori = []
        if logic0 or logic1:
            he_sens_ori.append(valid_sentence1)
            tmp = valid_sentence1.split(' ')
            tmp_out = ['she' if s=='he' or s=='He' else s for s in tmp]
            tmp_out2 = ['her' if s=='His' or s=='his' or s=='him' else s for s in tmp]
            valid_sentence0 = valid_sentence0.replace('他', '她')
            all_he_sens1.append(' '.join(tmp_out))
            all_he_sens0.append(valid_sentence0)
            # seen.add(current_name0)  

    with open(args.lan0_h_out, 'a') as f:
        for i in range(len(all_he_sens0)):
            f.write(all_he_sens0[i]+'1\n')
    with open(args.lan1_h_out, 'a') as f:
        for i in range(len(all_he_sens1)):
            f.write(all_he_sens1[i]+'1\n')
    print(len(all_he_sens0), len(all_he_sens1))

    if args.shuffle:        
        with open(args.shuffled_lan0_out, 'a') as f:
            for i in range(len(all_she_sens0)):
                f.write(all_she_sens0[i]+'0\n')
                f.write(all_he_sens0[3*i]+'1\n')
                f.write(all_he_sens0[3*i+1]+'1\n')
                f.write(all_he_sens0[3*i+2]+'1\n')

        with open(args.shuffled_lan1_out, 'a') as f:
            for i in range(len(all_she_sens0)):
                f.write(all_she_sens1[i]+'0\n')
                f.write(all_he_sens1[3*i]+'1\n')
                f.write(all_he_sens1[3*i+1]+'1\n')
                f.write(all_he_sens1[3*i+2]+'1\n')

        with open(args.shuffled_lan1_out_ori, 'a') as f:
            for i in range(len(all_she_sens0)):
                f.write(all_she_sens1[i]+'0\n')
                f.write(all_he_sens1[3*i]+'1\n')
                f.write(all_he_sens1[3*i+1]+'1\n')
                f.write(all_he_sens1[3*i+2]+'1\n')

        with open(args.shuffled_lan1_out_ori, 'a') as f:
            for i in range(len(all_she_sens0)):
                f.write(all_she_sens1[i]+'0\n')
                f.write(all_he_sens1[3*i]+'1\n')
                f.write(all_he_sens1[3*i+1]+'1\n')
                f.write(all_he_sens1[3*i+2]+'1\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('-lan0_s_in', type=str, help='the file that contains only female celebrities')
    parser.add_argument('-lan1_s_in',type=str, help='the file that contains only male celebrities')
    parser.add_argument('-lan0_h_in', type=str, help='the file that contains only female celebrities')
    parser.add_argument('-lan1_h_in',type=str, help='the file that contains only male celebrities')
    parser.add_argument('-lan0_s_out',type= str,  help="output files of language 1 gender-balanced sentence")
    parser.add_argument('-lan1_s_out',type = str,  help='output file of language 2 gender balanaced sentence')
    parser.add_argument('-lan0_h_out',type= str,  help="output files of language 1 gender-balanced sentence")
    parser.add_argument('-lan1_h_out',type = str,  help='output file of language 2 gender balanaced sentence')
    parser.add_argument('-shuffle', action='store_true')
    parser.add_argument('-shuffled_lan0_out', type=str)
    parser.add_argument('-shuffled_lan1_out', type=str)
    parser.add_argument('-shuffled_lan0_out_ori', type=str)
    parser.add_argument('-shuffled_lan1_out_ori', type=str)
    args = parser.parse_args()

    print(args)
    # combine(args)
    combine_switch_gender(args)
