import argparse
import preprocess.data_reader as data_reader
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Reshape
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
import meth_profiler as mp
import profile_generator as pg
import numpy as np
from os.path import exists
import random
from tensorflow.keras.layers import Conv2D, Input, concatenate
import preprocess.preprocess as preprocess

if not os.path.exists('./dump_files/'):
    os.makedirs('./dump_files/')
if not os.path.exists('./models/'):
    os.makedirs('./models/')
parser = argparse.ArgumentParser(description='AMPS')

parser.add_argument('-m', '--methylation_file', help='methylation file address', required=True)
parser.add_argument('-g', '--genome_assembly_file', help='genome sequence file address, must be in fasta format', required=True)
parser.add_argument('-c', '--context', help='context', required=True)
parser.add_argument('-ga', '--gene_file', help='gene annotation file address', required=False)
parser.add_argument('-ra', '--repeat_file', help='repeat annotation file address', required=False)
parser.add_argument('-iga', '--include_gene', help='does the predictor include the gene annotation in the input? True/False', required=False, default=False)
parser.add_argument('-ira', '--include_repeat', help='does the predictor include the repeat annotation in the input? True/False', required=False, default=False)
parser.add_argument('-tr', '--train_size', help='training dataset size, number of inputs for training', required=False, default=500000, type=int)
parser.add_argument('-ct', '--coverage_threshold', help='minimum number of reads for including a cytosine in the training dataset', required=False, default=10, type=int)
parser.add_argument('-on', '--organism_name', help='Organism name, for saving the files', required=False, default='sample_organism')
parser.add_argument('-mcs', '--memory_chunk_size', help='number of inputs in each memory load', required=False, default=1000, type=int)
args = parser.parse_args()

def input_maker(methylations,  datasize, window_size, organism_name, from_file, context, half_w, methylated = True):
    methylations = methylations.sort_values(["chr", "position"], ascending=(True, True))
    chrs_counts = methylations['chr'].value_counts()
    last_chr_pos = {}
    chrnums = list(chrs_counts.index)
    sum = 0
    for i in range(len(chrnums)):
        if i in chrs_counts.keys():
            last_chr_pos[i] = sum+chrs_counts[i]-1
            sum += chrs_counts[i]
    # last_chr_pos ==> {0: 5524, 1: 1042784, 2: 1713034, 3: 2550983, 4: 3205486, 5: 4145381, 6: 4153872}
    # methylations.iloc[2550983] => chr 3.0 position    23459763.0
    # methylations.iloc[2550984] => chr 4.0 position    1007
    methylations.insert(0, 'idx', range(0, len(methylations)))
    sub_methylations = methylations[methylations['context'] == context]
    if len(context) == 0:
        sub_methylations = methylations
    idxs = sub_methylations['idx']
    mlevels = methylations['mlevel']
    mlevels = np.asarray(mlevels)
    X = np.zeros((datasize, window_size))
    Y = np.zeros(datasize)
    avlbls = np.asarray(idxs)
    for lcp in list(last_chr_pos.values()):
        if lcp > 0 and lcp < len(mlevels) - window_size:
            avlbls = np.setdiff1d(avlbls, range(lcp-half_w, lcp+half_w))
    if methylated:
        filtered_avlbls = [x for x in avlbls if mlevels[x] > 0.5]
    else:
        filtered_avlbls = [x for x in avlbls if mlevels[x] <= 0.5]
    smple = random.sample(list(filtered_avlbls), min(datasize, len(filtered_avlbls)))
    count_errored = 0
    print('border conditions: ', np.count_nonzero(np.asarray(smple) < half_w))
    for index, p in enumerate(smple):
        try:
            X[index] = np.concatenate((mlevels[p-half_w: p], mlevels[p+1: p+half_w+1]), axis=0)
            Y[index] = 0 if mlevels[p] < 0.5 else 1
        except ValueError:
            print(index, p)
            count_errored += 1
    X = X.reshape(list(X.shape) + [1])
    print(count_errored, ' profiles faced error')
    return X, Y, methylations, smple

# This function is for testing, me: all the methylations data_frame, X_methylated: is input of the network which is a list of lists containing methylation of neighboring cytosins.
# p is the index in the me dataframe, i is the index of X_methylated.
# def check_check(me, X_methylated, p, i):
#     return np.all(np.concatenate([np.asarray(me.iloc[p - 10: p].mlevel), np.asarray(me.iloc[p+1: p+1+ 10].mlevel)]) == X_methylated[i,:,0])

def profiler(organism_name, methylations, context, datasize, sequences_onehot, annot_seqs_onehot, num_to_chr_dic,
             me_window_size=20, seq_window_size=3200, from_file=False, threshold=0.5, include_annot=True, include_repeat=True):
    half_w = int(me_window_size/2)
    X_methylated, Y_methylated, meth_me, smple_me = input_maker(methylations, int(datasize/2), me_window_size, organism_name, from_file, context, half_w, methylated=True)
    X_unmethylated, Y_unmethylated, meth_ume, smple_ume = input_maker(methylations, int(datasize/2), me_window_size, organism_name, from_file, context, half_w, methylated=False)
    profiles_me, targets_me = pg.get_profiles(meth_me, smple_me, sequences_onehot, annot_seqs_onehot, num_to_chr_dic, window_size=seq_window_size)
    X_seq_me, Y_me = pg.data_preprocess(profiles_me, targets_me, include_annot=include_annot | include_repeat)
    profiles_ume, targets_ume = pg.get_profiles(meth_ume, smple_ume, sequences_onehot, annot_seqs_onehot, num_to_chr_dic, window_size=seq_window_size)
    X_seq_ume, Y_ume = pg.data_preprocess(profiles_ume, targets_ume, include_annot=include_annot | include_repeat)
    return np.concatenate((X_methylated, X_unmethylated), axis=0), np.concatenate([X_seq_me, X_seq_ume], axis=0), np.concatenate((Y_methylated, Y_unmethylated), axis=0)

def run_experiment(X_seq, X_meth, Y, seq_cols, meth_window_size=20, seq_window_size= 3200,  test_percent=0.2):
    X_seq_train, X_seq_val, X_meth_train, X_meth_val, y_train, y_val = train_test_split(X_seq, X_meth, Y, test_size=test_percent, random_state=42)
    input_1 = Input(shape=(seq_window_size, seq_cols, 1), name='input_1')
    y = Conv2D(16, kernel_size=(1, seq_cols), padding='VALID', activation='relu')(input_1)
    y = Reshape((80, 40, 16), input_shape=(seq_window_size, 1, 16))(y)
    y = Conv2D(16, kernel_size=(5, 3), padding='VALID', activation='relu')(y)
    y = Flatten()(y)
    input_2 = Input(shape=(meth_window_size), name='input_2')
    x = Dense(meth_window_size, activation='relu', input_shape=((meth_window_size,1)))(input_2)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    combined = concatenate([x, y])
    x = Dense(16, activation='relu')(combined)
    x = Dropout(0.5)(x)
    x = Dense(8, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    opt = SGD(lr=0.001)
    model = tf.keras.models.Model(inputs=[input_1, input_2], outputs=x)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy'])
    model.fit([X_seq_train, X_meth_train], y_train, batch_size=32, epochs=20, verbose=1, validation_data=([X_seq_val, X_meth_val], y_val))
    model_tag = str(organism_name) + str(args.context) + '_combo' + '.mdl'
    print('model_saved in ./models directory with name:' + model_tag)
    model.save('./models/' + model_tag)

include_gene = args.include_gene == 'True'
include_repeat = args.include_repeat == 'True'
if include_gene and len(args.gene_file) == 0:
    print('Enter the gene annotation file address. The gene annotation file must be provided when the include gene annotation is True')
if include_repeat and len(args.repeat_file) == 0:
    print('Enter the repeat annotation file address. The repeat annotation file must be provided when the include repeat annotation is True')

organism_name = args.organism_name
sequences = data_reader.readfasta(args.genome_assembly_file)
print('genome sequence assembly is loaded...')
methylations_, num_to_chr_dic_ = data_reader.get_methylations(args.methylation_file,  args.coverage_threshold, context=args.context)
print('methylation level is loaded for ' + args.context + ' context ...' + str(len(methylations_)))
annot_seqs_onehot = []
if include_gene:
    annot_df = data_reader.read_annot(args.gene_file)
    annot_tag = ''
    annot_types = preprocess.get_annot_types(annot_df)
    print('list of annotated functional elements:' + str(annot_types))
    if len(annot_types) > 10:
        print('Too many functional elements. Shrink the annotation file. Keep less than 10 functional elements')
        exit()
    for at in annot_types:
        annot_subset = preprocess.subset_annot(annot_df, at)
        annot_str = preprocess.make_annotseq_dic(organism_name, at, annot_subset, sequences, from_file=False)
        annot_seqs_onehot.append(annot_str)
        annot_tag += at
if include_repeat:
    annot_df = data_reader.read_annot(args.repeat_file)
    sequences = data_reader.readfasta(args.genome_assembly_file)
    annot_str = preprocess.make_annotseq_dic(organism_name, 'repeat', annot_df, sequences, from_file=True, strand_spec=False)
    annot_seqs_onehot.append(annot_str)

sequences_onehot = preprocess.convert_assembely_to_onehot(organism_name, sequences, from_file=False)
if not include_gene:
    del annot_seqs_onehot
    annot_seqs_onehot = []
PROFILE_ROWS = 3200
PROFILE_COLS = 4
if include_gene:
    PROFILE_COLS = PROFILE_COLS + 2*len(annot_types)
if include_repeat:
    PROFILE_COLS = PROFILE_COLS + 1

X_meth, X_seq, Y = profiler(args.organism_name, methylations_, args.context, args.train_size, sequences_onehot, annot_seqs_onehot, num_to_chr_dic_)
run_experiment(X_seq, X_meth, Y, PROFILE_COLS, meth_window_size=20, seq_window_size=3200,  test_percent=0.2)
