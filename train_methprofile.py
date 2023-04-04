import argparse
import preprocess.data_reader as data_reader
import os
from tensorflow import keras
from tensorflow.keras.layers import Activation,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Reshape
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
import meth_profiler as mp

if not os.path.exists('./dump_files/'):
    os.makedirs('./dump_files/')
if not os.path.exists('./models/'):
    os.makedirs('./models/')
parser = argparse.ArgumentParser(description='AMPS')

parser.add_argument('-m', '--methylation_file', help='methylation file address', required=True)
parser.add_argument('-c', '--context', help='context', required=True)
parser.add_argument('-tr', '--train_size', help='training dataset size, number of inputs for training', required=False, default=500000, type=int)
parser.add_argument('-ws', '--window_size', help='window size, number of including nucleutides in a window.', required=False, default=20, type=int)
parser.add_argument('-ct', '--coverage_threshold', help='coverage_threshold, minimum number of reads for including a cytosine in the training dataset', required=False, default=10, type=int)
parser.add_argument('-on', '--organism_name', help='Organism name, for saving the files...', required=False, default='sample_organism')


args = parser.parse_args()

# args = argparse.Namespace()
# args.methylation_file = './sample/sample_methylations_train.txt'
# args.context = 'CG'
# args.train_size = 50000
# args.window_size = 20
# args.coverage_threshold = 10
# args.organism_name = 'sample_organism'

def run_experiment(organism_name, X, Y, window_size=20, val_percent=0.2):
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=val_percent, random_state=None)
    model = Sequential()
    model.add(Dense(window_size, activation='relu', input_shape=((window_size,1))))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.001)
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=0, validation_data=(x_val, y_val))
    model_tag = str(organism_name) + str(args.context) + '_methprofile' + '.mdl'
    print('model_saved in ./models directory with name:' + model_tag)
    model.save('./models/' + model_tag)

#gets methylation dataframe containing all the available cytosines.
methylations, num_to_chr_dic = data_reader.get_methylations(args.methylation_file,  args.coverage_threshold, context='')
X, Y = mp.profiler(methylations, args.context, args.train_size, window_size=args.window_size)
run_experiment(args.organism_name, X, Y, window_size=args.window_size, val_percent=0.1)
