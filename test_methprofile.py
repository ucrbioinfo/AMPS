import argparse
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import pandas as pd

if not os.path.exists('./dump_files/'):
    os.makedirs('./dump_files/')
if not os.path.exists('./models/'):
    os.makedirs('./models/')
if not os.path.exists('./output/'):
    os.makedirs('./output/')
parser = argparse.ArgumentParser(description='')

parser.add_argument('-p', '--prfiles_address', help='address to the file containing the cytosine profiles. a tab seperated file, each row is the methylation level of neighbouring Cytosines', required=True)
parser.add_argument('-mdl', '--model_address', help='trained model address', required=True)
parser.add_argument('-on', '--organism_name', help='Organism name, for saving the files...', required=False, default='sample_organism')

args = parser.parse_args()

args = argparse.Namespace()
args.prfiles_address = './sample/sample_meth_profile_test.txt'
args.model_address = './models/sample_organismCG_methprofile.mdl'
args.organism_name = 'sample_organism'


model = keras.models.load_model(args.model_address)

X = pd.read_table(args.prfiles_address)
X = X.to_numpy()
X = X.reshape(list(X.shape) + [1])

Y = model.predict(X)

np.savetxt('./output/' + args.organism_name+'_methprofile.npy', Y.round(), delimiter=' ', fmt='%d')
print('results saved in ./output/'+ args.organism_name+'_' + args.context + '.npy')
