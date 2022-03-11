import argparse
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
parser = argparse.ArgumentParser(description='This is a demo script by nixCraft.')

parser.add_argument('-p', '--prfiles_address', help='address to the file containing the cytosine profiles. a tab seperated file, each row is the methylation level of neighbouring Cytosines', required=True)
parser.add_argument('-mdl', '--model_address', help='trained model address', required=True)
parser.add_argument('-on', '--organism_name', help='Organism name, for saving the files...', required=False, default='sample_organism')


args = parser.parse_args()

model = keras.models.load_model(args.model_address)

X = pd.read_table(args.prfiles_address)
X = X.to_numpy()
X = X.reshape(list(X.shape) + [1])

Y = model.predict(X)

np.save('./output/' + args.organism_name+'_' + args.context + '_methprofile.npy', Y.round())
print('results saved in ./output/'+ args.organism_name+'_' + args.context + '.npy')
