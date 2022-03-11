import argparse
import preprocess.data_reader as data_reader
from tensorflow import keras
import os
import meth_profiler as mp
import numpy as np

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
parser.add_argument('-te', '--test_size', help='testing dataset size, number of inputs for training', required=False, default=50000, type=int)
parser.add_argument('-ws', '--window_size', help='window size, number of including nucleutides in a window. It has to be similar to the training set window-size', required=False, default=3200, type=int)
parser.add_argument('-ct', '--coverage_threshold', help='coverage_threshold, minimum number of reads for including a cytosine in the training dataset', required=False, default=10, type=int)
parser.add_argument('-on', '--organism_name', help='Organism name, for saving the files...', required=False, default='sample_organism')


args = parser.parse_args()

model = keras.models.load_model(args.model_address)

methylations, num_to_chr_dic = data_reader.get_methylations(args.methylation_file,  args.coverage_threshold, context=args.context)
X, Y = mp.profiler(methylations, args.context, args.train_size, window_size=args.window_size)

Y = model.predict(X)

np.save('./output/' + args.organism_name+'_' + args.context + '_methprofile.npy', Y.round())
print('results saved in ./output/'+ args.organism_name+'_' + args.context + '.npy')
