import argparse
import preprocess.data_reader as data_reader
import preprocess.preprocess as preprocess
import os
import random
import profile_generator as pg
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import Activation,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Reshape

if not os.path.exists('./dump_files/'):
    os.makedirs('./dump_files/')
if not os.path.exists('./models/'):
    os.makedirs('./models/')
if not os.path.exists('./output/'):
    os.makedirs('./output/')
parser = argparse.ArgumentParser(description='This is a demo script by nixCraft.')

parser.add_argument('-m', '--methylation_file', help='methylation file address', required=True)
parser.add_argument('-mdl', '--model_address', help='trained model address', required=True)
parser.add_argument('-g', '--genome_assembly_file', help='genome sequence file address, must be in fasta format', required=True)
parser.add_argument('-c', '--context', help='context', required=True)
parser.add_argument('-a', '--annotation_file', help='annotation file', required=False)
parser.add_argument('-ia', '--include_annotation', help='does the predictor include the annotation in the input? True/False, It has to be similar to training dataset', required=False, default=False)
parser.add_argument('-te', '--test_size', help='testing dataset size, number of inputs for training', required=False, default=50000, type=int)
parser.add_argument('-ws', '--window_size', help='window size, number of including nucleutides in a window. It has to be similar to the training set window-size', required=False, default=3200, type=int)
parser.add_argument('-ct', '--coverage_threshold', help='coverage_threshold, minimum number of reads for including a cytosine in the training dataset', required=False, default=10, type=int)
parser.add_argument('-on', '--organism_name', help='Organism name, for saving the files...', required=False, default='sample_organism')


args = parser.parse_args()

include_annot = args.include_annotation == 'True'
if include_annot and len(args.annotation_file) == 0:
    print('Enter the annotation file address. The annotation file must be provided when the include annotation is True')
model = keras.models.load_model(args.model_address)

organism_name = args.organism_name
sequences = data_reader.readfasta(args.genome_assembly_file)
print('genome sequence assembly is loaded...')
methylations, num_to_chr_dic = data_reader.get_methylations(args.methylation_file,  args.coverage_threshold, context=args.context)
print('methylation level is loaded for ' + args.context + ' context ...')
if include_annot:
    annot_df = data_reader.read_annot(args.annotation_file)
    annot_seqs_onehot = []
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
sequences_onehot = preprocess.convert_assembely_to_onehot(organism_name, sequences, from_file=False)
if not include_annot:
    del annot_seqs_onehot
    annot_seqs_onehot = []

test_profiles, test_targets = pg.get_profiles(methylations, range(len(methylations)), sequences_onehot, annot_seqs_onehot, num_to_chr_dic, window_size=args.window_size)
x_test, y_test = pg.data_preprocess(test_profiles, test_targets, include_annot=include_annot)

y_pred = model.predict(x_test)

np.save('./output/' + organism_name+'_' + args.context + '.npy', y_pred)
