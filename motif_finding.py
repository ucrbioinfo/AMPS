from tensorflow import keras
import os
import argparse
import motif_analysis as motif_analysis
import preprocess.preprocess as preprocess
import preprocess.data_reader as data_reader

if not os.path.exists('./motifs/'):
    os.makedirs('./motifs/')

parser = argparse.ArgumentParser(description='AMPS')

parser.add_argument('-mdl', '--model_address', help='trained model address', required=True)
parser.add_argument('-seqs', '--sequence_file', help='fasta file containing the sequences which you want to find the motifs in them.', required=True)
parser.add_argument('-ms', '--motif_size', help='size of motifs to search in the input set', required=False, default=50)
parser.add_argument('-o', '--output', help='output_file_name', required=False, default='sample_motifs.fa')


args = parser.parse_args()

model = keras.models.load_model(args.model_address)
seqs_oh = preprocess.convert_fasta_file_to_onehot(data_reader.readfasta(args.sequence_file))
motif_analysis.save_motif_fasta_files(model, seqs_oh, args.motif_size, './motifs/' + args.output)
print('Motifs were saved to ./motifs/' + args.output)




