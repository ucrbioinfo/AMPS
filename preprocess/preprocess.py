import numpy as np
import constants as constants
import pickle
from os import path
import random
import pandas as pd

def subset_annot(annot_df, type):
    genes_df = annot_df[annot_df['type'] == type]
    genes_df = genes_df.reset_index(drop=True)
    genes_df = genes_df.sort_values(['chr', 'start'], ascending=(True, True))
    genes_df = genes_df.reset_index(drop=True)
    return genes_df[['chr', 'strand', 'start', 'end']]

def get_annot_types(annot_df):
    types = annot_df['type'].value_counts()
    functional_elements = []
    for k in types.keys():
        functional_elements.append(k)
    return functional_elements


def make_annotseq_dic(organism_name, annot_tag, annot_subset_df, sequences, from_file=False, strand_spec=True):
    fn = './dump_files/' + organism_name + '_' + annot_tag+'_annot_seqs.pkl'
    if from_file and path.exists(fn):
        return load_dic(fn)
    annot_seqs = {}
    count = 0
    for chr in sequences.keys():
        annot_seq_p = np.zeros((len(sequences[chr]), 1), dtype='short')
        annot_seq_n = np.zeros((len(sequences[chr]), 1), dtype='short')
        annot_df_chr_subset = annot_subset_df[annot_subset_df['chr'] == chr]
        for index, row in annot_df_chr_subset.iterrows():
            if row['strand'] == '+' or not strand_spec:
                annot_seq_p[int(row['start'] - 1): int(row['end'] - 1)] = 1
            else:
                annot_seq_n[int(row['start'] - 1): int(row['end'] - 1)] = 1
            if count % int(len(annot_subset_df)/10) == 0:
                print(str(int(count * 100/len(annot_subset_df))) + '%')
            count += 1
        if strand_spec:
            annot_seqs[chr] = np.concatenate([annot_seq_p, annot_seq_n], axis=1)
        else:
            annot_seqs[chr] = annot_seq_p
    save_dic(fn, annot_seqs)
    return annot_seqs

def shrink_methylation(methylations, include_context = False):
    chr_ndarray = np.asarray(methylations['chr'])
    positions_ndarray = np.asarray(methylations['position'])
    mlevels_ndarray = np.asarray(methylations['meth']/(methylations['meth']+methylations['unmeth']))
    chr_to_num_dic = {}
    unique_chrs, unique_indices, chr_ndarray = np.unique(chr_ndarray, return_inverse=True, return_index=True)
    for i in range(len(unique_chrs)):
        chr_to_num_dic[chr_ndarray[unique_indices[i]]] = unique_chrs[i]
    chr_ndarray = chr_ndarray.astype('short')
    if include_context:
        methylations = pd.DataFrame({'chr': chr_ndarray, 'position': positions_ndarray, 'mlevel': mlevels_ndarray, 'context': methylations['context']})
    else:
        methylations = pd.DataFrame({'chr': chr_ndarray, 'position': positions_ndarray, 'mlevel': mlevels_ndarray})

    methylations['chr'] = methylations['chr'].astype(int)
    methylations['position'] = methylations['position'].astype(int)
    methylations['mlevel'] = methylations['mlevel'].astype(float)
    return methylations, chr_to_num_dic



def replace_values_nparray(np_array, chro_to_num_dic):
    from_values = list(chro_to_num_dic.keys())
    to_values = []
    for k in from_values:
        to_values.append(chro_to_num_dic[k])
    sort_idx = np.argsort(from_values)
    idx = np.searchsorted(from_values, np_array,sorter= sort_idx)
    out = to_values[sort_idx][idx]
    return out

def replace_values_nparray(np_array, chro_to_num_dic):
    out = [chro_to_num_dic[i] for i in np_array]
    return out


def make_meth_string(organism_name, methylations, sequences, coverage_thrshld, from_file = False):
    fn = './dump_files/' + organism_name + '_meth_seq.pkl'
    if from_file and path.exists(fn):
        return load_dic(fn)

    methylations['mlevel'] = methylations['meth']/ (methylations['meth'] + methylations['unmeth'])
    methylations['coverage'] = methylations['meth'] + methylations['unmeth']
    methylations['mlevel'] = methylations['mlevel'].fillna(0)

    methylations.loc[(methylations.mlevel == 0), 'mlevel'] = constants.NON_METH_TAG
    methylations.loc[(methylations.coverage < coverage_thrshld), 'mlevel'] = 0
    #methylations.loc[(methylations.strand == '-'),'mlevel'] = -1 * methylations.mlevel

    meth_seq = {}
    for chr in sequences.keys():
        meths = np.zeros(len(sequences[chr]))
        meth_subset = methylations[methylations['chr'] == chr]
        meths[[meth_subset['position'] - 1]] = meth_subset['mlevel']
        meth_seq[chr] = meths
    save_dic(fn, meth_seq)
    return meth_seq

def convert_seq_to_onehot(seq):
    seq_np = np.asarray(list(seq))
    seq_np = np.char.lower(seq_np)
    idx = np.char.equal(seq_np, np.asarray(['a'] * len(seq)))
    As = np.zeros((len(seq), 1), dtype='short')
    As[idx] = 1
    idx = np.char.equal(seq_np, np.asarray(['c'] * len(seq)))
    Cs = np.zeros((len(seq), 1), dtype='short')
    Cs[idx] = 1
    idx = np.char.equal(seq_np, np.asarray(['g'] * len(seq)))
    Gs = np.zeros((len(seq), 1), dtype='short')
    Gs[idx] = 1
    idx = np.char.equal(seq_np, np.asarray(['t'] * len(seq)))
    Ts = np.zeros((len(seq), 1), dtype='short')
    Ts[idx] = 1
    res = np.concatenate([As, Cs, Gs, Ts], axis=1)
    return res

def convert_assembely_to_onehot(organism_name, sequences, from_file=False):
    fn = './dump_files/' + organism_name + '_sequences_onehot.pkl'
    if from_file and path.exists(fn):
        return load_dic(fn)
    one_hots = {}
    for key in sequences.keys():
        one_hots[key] = convert_seq_to_onehot(sequences[key])
    save_dic(fn, one_hots)
    return one_hots



def methylations_subseter(methylations, window_size):
    methylations['m_idx'] = range(len(methylations))
    methylations_subset = methylations[methylations['position'] > window_size * 10]
    methylated = methylations_subset[methylations_subset['mlevel'] > 0.5]
    methylated = methylated.reset_index(drop=True)
    unmethylated = methylations_subset[methylations_subset['mlevel'] < 0.5]
    unmethylated = unmethylated.reset_index(drop=True)
    methylated = list(methylated['m_idx'])
    random.shuffle(methylated)
    unmethylated = list(unmethylated['m_idx'])
    random.shuffle(unmethylated)
    return methylated, unmethylated

#returns two lists of shuffled numbers with the same size
def methylations_sampler(methylated_size, unmethylated_size):
    shuffled_m = range(methylated_size)
    random.shuffle(shuffled_m)
    shuffled_um = range(unmethylated_size)
    random.shuffle(shuffled_um)
    if methylated_size > unmethylated_size:
        shuffled_m = shuffled_m[:len(shuffled_um)]
    else:
        shuffled_um = shuffled_um[:len(shuffled_m)]
    return shuffled_m, shuffled_um

def seperate_methylations(organism_name, methylations, test_ratio = 0.1, from_file=False):

    fn_train = './dump_files/'+organism_name+'_methylations_train.csv'
    fn_test = './dump_files/'+organism_name+'_methylations_test.csv'
    if from_file and path.exists(fn_test) and path.exists(fn_train):
        methylations_train = pd.read_csv('./dump_files/'+organism_name+'_methylations_train.csv', header=0)
        methylations_test = pd.read_csv('./dump_files/'+organism_name+'_methylations_test.csv', header=0)
        return methylations_train, methylations_test
    idxs = np.random.permutation(len(methylations))
    splitter = int(len(methylations)*(1-test_ratio))
    methylations_train = methylations.loc[idxs[0:splitter]]
    methylations_train = methylations_train.reset_index(drop=True)
    methylations_test = methylations.loc[idxs[splitter:len(methylations)]]
    methylations_test = methylations_test.reset_index(drop=True)
    methylations_train.to_csv('./dump_files/'+organism_name+'_methylations_train.csv', header=True, index=False)
    methylations_test.to_csv('./dump_files/'+organism_name+'_methylations_test.csv', header=True, index=False)
    return methylations_train, methylations_test

def cpgenie_preprocess(X, Y):
    X = np.delete(X, range(4, X.shape[2]), 2)
    Y = np.asarray(pd.cut(Y, bins=2, labels=[0, 1], right=False))
    b = np.zeros((Y.size, Y.max()+1))
    b[np.arange(Y.size), Y] = 1
    Y = b
    X = X.reshape(list(X.shape) + [1])
    X = np.swapaxes(X, 1, 2)
    return X, Y

def save_dic(file_name, dic):
    f = open(file_name, "wb")
    pickle.dump(dic, f)
    f.close()

def load_dic(file_name):
    f = open(file_name, "rb")
    res = pickle.load(f)
    f.close()
    return res
