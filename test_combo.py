import argparse
import preprocess.data_reader as data_reader
import preprocess.preprocess as preprocess
import os
import random
import profile_generator as pg
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

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
parser.add_argument('-p', '--prfiles_address', help='address to the file containing the cytosine profiles. a tab seperated file, each row is the methylation level of neighbouring Cytosines', required=True)
parser.add_argument('-ga', '--gene_file', help='gene annotation file', required=False)
parser.add_argument('-ra', '--repeat_file', help='repeat annotation file', required=False)
parser.add_argument('-iga', '--include_gene', help='does the predictor include the gene annotation in the input? True/False, It has to be similar to training dataset', required=False, default=False)
parser.add_argument('-ira', '--include_repeat', help='does the predictor include the repeat annotation in the input? True/False, It has to be similar to training dataset', required=False, default=False)
parser.add_argument('-ws', '--window_size', help='window size, number of including nucleutides in a window. It has to be similar to the training set window-size', required=False, default=3200, type=int)
parser.add_argument('-on', '--organism_name', help='Organism name, for saving the files...', required=False, default='sample_organism')


args = parser.parse_args()


# args = argparse.Namespace()
# args.methylation_file = './sample/sample_methylations_test.txt'
# args.genome_assembly_file = './sample/sample_seq.fasta'
# args.prfiles_address = './sample/sample_meth_profile_test.txt'
# args.model_address = './models/sample_organismCG_combo.mdl'
# args.gene_file = './sample/sample_gene_annotation.txt'
# args.repeat_file = './sample/sample_repeat_annotation.txt'
# args.include_gene = 'True'
# args.include_repeat = 'True'
# args.context = 'CG'
# args.train_size = 50000
# args.coverage_threshold = 10
# args.organism_name = 'sample_organism'
# args.memory_chunk_size = 1000

include_gene = args.include_gene == 'True'
include_repeat = args.include_repeat == 'True'
if include_gene and len(args.gene_file) == 0:
    print('Enter the gene annotation file address. The gene annotation file must be provided when the include gene annotation is True')
if include_repeat and len(args.repeat_file) == 0:
    print('Enter the repeat annotation file address. The repeat annotation file must be provided when the include repeat annotation is True')
model = keras.models.load_model(args.model_address)

organism_name = args.organism_name
sequences = data_reader.readfasta(args.genome_assembly_file)
print('genome sequence assembly is loaded...')
methylations, num_to_chr_dic = data_reader.get_methylations(args.methylation_file,  -1, '', contain_targets=False)
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
sequences_onehot = preprocess.convert_assembely_to_onehot(organism_name, sequences, from_file=False)
if not include_gene:
    del annot_seqs_onehot
    annot_seqs_onehot = []

if include_repeat:
    annot_df = data_reader.read_annot(args.repeat_file)
    sequences = data_reader.readfasta(args.genome_assembly_file)
    annot_str = preprocess.make_annotseq_dic(organism_name, 'repeat', annot_df, sequences, from_file=True, strand_spec=False)
    annot_seqs_onehot.append(annot_str)

#methylations['mlevel'] = methylations['mlevel'].fillna(0)
test_profiles, _ = pg.get_profiles(methylations, range(len(methylations)), sequences_onehot, annot_seqs_onehot, num_to_chr_dic, window_size=3200, contain_targets=False)
x_test_seq, _ = pg.data_preprocess(test_profiles, None, include_annot=include_gene | include_repeat, contain_targets=False)

x_test_meth = pd.read_csv(args.prfiles_address, header=None, sep='\t')
x_test_meth = x_test_meth.to_numpy()
x_test_meth = x_test_meth.reshape(list(x_test_meth.shape) + [1])


y_pred = model.predict([x_test_seq, x_test_meth])

np.savetxt('./output/' + organism_name + '.txt', y_pred.round().astype(int), delimiter=' ', fmt='%d')
print('results saved in ./output/' + organism_name + '.txt')
