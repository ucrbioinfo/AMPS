from Bio import SeqIO
import pandas as pd
import pickle
from os import path
import numpy as np
import constants
import preprocess.preprocess as preprocess

def readfasta(address):
    recs = SeqIO.parse(address, "fasta")
    sequences = {}
    for chro in recs:
        sequences[chro.id] = chro.seq
    for i in sequences.keys():
        sequences[i] = sequences[i].upper()
    return sequences

def read_annot(address, chromosomes = None):
    annot_df = pd.read_table(address, sep='\t', comment='#')
    annot_df.columns = ['chr', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']
    if chromosomes != None:
        annot_chrs = annot_df.chr.unique()
        for chr in annot_chrs:
            if chr not in chromosomes:
                annot_df = annot_df[annot_df['chr'] != chr]
    return annot_df

def get_methylations(address, coverage_threshold, context=None):
    methylations = read_methylations(address, context, coverage_threshold=coverage_threshold)
    include_context = True
    if context == None:
        include_context = False
    methylations, num_to_chr_dic = preprocess.shrink_methylation(methylations, include_context=include_context)
    return methylations, num_to_chr_dic

def read_methylations(address, context, coverage_threshold = 10):
    methylations = pd.read_table(address, header=None)
    methylations.columns = ['chr', 'position', 'strand', 'meth', 'unmeth', 'context', 'three']
    methylations.drop(['three'], axis=1)
    if len(context) != 0:
        methylations = methylations[methylations['context'] == context]
    methylations = methylations[methylations['meth'] + methylations['unmeth'] > coverage_threshold]
    if len(context) != 0:
        methylations.drop(['context'], axis=1)
    methylations.drop(['strand'], axis=1)
    methylations = methylations.reset_index(drop=True)
    return methylations

def make_meth_string(organism_name, methylations, sequences, coverage_thrshld, from_file=False):
    fn = './dump_files/' + organism_name + '_meth_seq.pkl'
    if from_file and path.exists(fn):
        return load_dic(fn)
    methylations['mlevel'] = methylations['meth']/(methylations['meth'] + methylations['unmeth'])
    methylations['coverage'] = methylations['meth'] + methylations['unmeth']
    methylations['mlevel'] = methylations['mlevel'].fillna(0)
    methylations.loc[(methylations.mlevel == 0), 'mlevel'] = constants.NON_METH_TAG
    methylations.loc[(methylations.coverage < coverage_thrshld), 'mlevel'] = 0
    methylations.loc[(methylations.strand == '-'),'mlevel'] = -1 * methylations.mlevel
    meth_seq = {}
    count = 0
    for chr in sequences.keys():
        meths = np.zeros(len(sequences[chr]))
        meth_subset = methylations[methylations['chr'] == chr]
        meths[[meth_subset['position'] - 1]] = meth_subset['mlevel']
        meth_seq[chr] = meths
        count+=1
        print(count, len(sequences.keys()))
    save_dic(fn, meth_seq)
    return meth_seq

def make_context_string(organism_name, methylations, sequences, from_file=False):
    fn = './dump_files/' + organism_name + '_context_seq.pkl'
    if from_file and path.exists(fn):
        return load_dic(fn)
    methylations['context_id'] = 1
    methylations.loc[(methylations.context == 'CHG'), 'context_id'] = 2
    methylations.loc[(methylations.context == 'CHH'), 'context_id'] = 3
    context_seq = {}
    count = 0
    for chr in sequences.keys():
        contexts = np.zeros(len(sequences[chr]))
        meth_subset = methylations[methylations['chr'] == chr]
        contexts[[meth_subset['position'] - 1]] = meth_subset['context_id']
        context_seq[chr] = contexts
        count+=1
        print(count, len(sequences.keys()))
    save_dic(fn, context_seq)
    return context_seq

def subset_annot(annot_df, type):
    genes_df = annot_df[annot_df['type'] == type]
    genes_df = genes_df.reset_index(drop=True)
    genes_df = genes_df.sort_values(['chr', 'start'], ascending=(True, True))
    genes_df = genes_df.reset_index(drop=True)
    return genes_df[['chr', 'strand', 'start', 'end']]

def save_dic(file_name, dic):
    f = open(file_name, "wb")
    pickle.dump(dic, f)
    f.close()

def load_dic(file_name):
    f = open(file_name, "rb")
    res = pickle.load(f)
    f.close()
    return res
