from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



#This method gets for an organism and context, mekes
def get_profiles(methylations, sample_set, sequences_onehot, annot_seqs_onehot, num_to_chr_dic, window_size=3200):
    log = len(sample_set) > 50000
    log = False
    boundary_cytosines = 0
    profiles = np.zeros([len(sample_set), window_size, 4 + 2*len(annot_seqs_onehot)], dtype='short')
    targets = np.zeros(len(sample_set), dtype='short')
    total = len(sample_set)
    count = 0
    start = datetime.now()
    for index, position in enumerate(sample_set):
        row = methylations.iloc[position]
        center = int(row['position'] - 1)
        chro = num_to_chr_dic[row['chr']]
        targets[index] = round(float(row['mlevel']))
        try:
            profiles[index] = get_window_seqgene_df(sequences_onehot, annot_seqs_onehot, chro, center, window_size)
        except:
            boundary_cytosines += 1
        if count % int(total/10) == 0:
            now = datetime.now()
            seconds = (now - start).seconds
            if log:
                print(str(int(count * 100/total)) + '%' + ' in ' + str(seconds) +' seconds')
        count += 1
    if log:
        print(str(boundary_cytosines) + ' boundary cytosines are ignored')
        print(datetime.now())
    return profiles, targets



def get_window_seqgene_df(sequences_df, annot_seq_df_list, chro, center, window_size):
    profile_df = sequences_df[chro][center - int(window_size/2): center + int(window_size/2)]
    for i in range(len(annot_seq_df_list)):
        profile_df = np.concatenate([profile_df, annot_seq_df_list[i][chro][center - int(window_size/2): center + int(window_size/2)]], axis=1)
    return profile_df

def data_preprocess(X, Y, include_annot=False):
    if not include_annot:
        X = np.delete(X, range(4, X.shape[2]), 2)
    X = X.reshape(list(X.shape) + [1])
    Y = np.asarray(pd.cut(Y, bins=2, labels=[0, 1], right=False))
    return X, Y

def split_data(X, Y, pcnt=0.1):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=pcnt, random_state=None)
    return x_train, x_test, y_train, y_test

