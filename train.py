import argparse
import preprocess.data_reader as data_reader
import preprocess.preprocess as preprocess
import os
import random
import profile_generator as pg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Reshape

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
parser.add_argument('-ws', '--window_size', help='window size, number of including nucleutides in a window.', required=False, default=3200, type=int)
parser.add_argument('-ct', '--coverage_threshold', help='minimum number of reads for including a cytosine in the training dataset', required=False, default=10, type=int)
parser.add_argument('-on', '--organism_name', help='Organism name, for saving the files', required=False, default='sample_organism')
parser.add_argument('-mcs', '--memory_chunk_size', help='number of inputs in each memory load', required=False, default=1000, type=int)


args = parser.parse_args()

include_gene = args.include_gene == 'True'
include_repeat = args.include_repeat == 'True'
if include_gene and len(args.gene_file) == 0:
    print('Enter the gene annotation file address. The gene annotation file must be provided when the include gene annotation is True')
if include_repeat and len(args.repeat_file) == 0:
    print('Enter the repeat annotation file address. The repeat annotation file must be provided when the include repeat annotation is True')

organism_name = args.organism_name
sequences = data_reader.readfasta(args.genome_assembly_file)
print('genome sequence assembly is loaded...')
methylations, num_to_chr_dic = data_reader.get_methylations(args.methylation_file,  args.coverage_threshold, context=args.context)
print('methylation level is loaded for ' + args.context + ' context ...' + str(len(methylations)))
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
model = Sequential()
model.add(Conv2D(16, kernel_size=(1, PROFILE_COLS), activation='relu', input_shape=(PROFILE_ROWS, PROFILE_COLS, 1)))
model.add(Reshape((80, 40, 16), input_shape=(PROFILE_ROWS, 1, 16)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
print('model processed')
opt = tf.keras.optimizers.SGD(lr=0.01)
model.compile(loss=keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy'])
methylated_train, unmethylated_train = preprocess.methylations_subseter(methylations, 3200)
ds_size = min(len(methylated_train), len(unmethylated_train))
print('ds_size', ds_size)
x_train_sz = 0
step = args.train_size
if ds_size * 2 < step:
    step = (ds_size * 2) - 2
print('start fitting ... ' + 'dataset size = ' + str(step))
print(step, args.memory_chunk_size)
gpu_available = tf.test.is_gpu_available()
for chunk in range(0, int(step/2), args.memory_chunk_size):
    if chunk+args.memory_chunk_size > int(step/2):
        sample_set = methylated_train[chunk:int(step/2)]+unmethylated_train[chunk:int(step/2)]
    else:
        sample_set = methylated_train[chunk:chunk+args.memory_chunk_size]+unmethylated_train[chunk:chunk+args.memory_chunk_size]
    if len(sample_set) == 0:
        continue
    random.shuffle(sample_set)
    profiles, targets = pg.get_profiles(methylations, sample_set, sequences_onehot, annot_seqs_onehot, num_to_chr_dic, window_size=3200)
    X, Y = pg.data_preprocess(profiles, targets, include_annot=include_gene | include_repeat)
    x_train, x_val, y_train, y_val = pg.split_data(X, Y, pcnt=0.1)
    x_train_sz += len(x_train)
    if gpu_available:
        with tf.device('/device:GPU:0'):
            model.fit(x_train, y_train, batch_size=32, epochs=45, verbose=0, validation_data=(x_val, y_val))
    else:
        model.fit(x_train, y_train, batch_size=32, epochs=45, verbose=0, validation_data=(x_val, y_val))
    del x_train, y_train
print('trained with ' + str(x_train_sz) + ' entries')
ia_tag = ''
if include_gene:
    ia_tag += 'Annot'
if include_repeat:
    ia_tag += 'Repeat'
model_tag = str(organism_name) + str(args.context) + ia_tag + '.mdl'
print('model_saved in ./models directory with name:' + model_tag)
model.save('./models/' + model_tag)
