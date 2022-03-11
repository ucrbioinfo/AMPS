import mrcnn, cpgenie
import argparse
import preprocess.data_reader as data_reader
import preprocess.preprocess as preprocess

parser = argparse.ArgumentParser(description='AMPS')

parser.add_argument('-m', '--methylation_file', help='methylation file address', required=True)
parser.add_argument('-cls', '--classifier', help='name of classifier: cpgenie/mrcnn', required=True)
parser.add_argument('-g', '--genome_assembly_file', help='genome sequence file address, must be in fasta format', required=True)
parser.add_argument('-c', '--context', help='context', required=True)
parser.add_argument('-tr', '--train_size', help='training dataset size, number of inputs for training', required=False, default=500000, type=int)
parser.add_argument('-ct', '--coverage_threshold', help='coverage_threshold, minimum number of reads for including a cytosine in the training dataset', required=False, default=10, type=int)
parser.add_argument('-on', '--organism_name', help='Organism name, for saving the files...', required=False, default='sample_organism')
parser.add_argument('-mcs', '--memory_chunk_size', help='number of inputs in each memory load', required=False, default=1000, type=int)

args = parser.parse_args()

classifier = args.classifier


organism_name = args.organism_name
sequences = data_reader.readfasta(args.genome_assembly_file)
print('genome sequence assembly is loaded...')
methylations, num_to_chr_dic = data_reader.get_methylations(args.methylation_file,  args.coverage_threshold, context=args.context)
print('methylation level is loaded for ' + args.context + ' context ...')
sequences_onehot = preprocess.convert_assembely_to_onehot(organism_name, sequences, from_file=False)



if classifier == 'mrcnn':
    mrcnn.run_experiment(methylations, sequences_onehot, num_to_chr_dic, data_size=args.train_size)
elif classifier == 'cpgenie':
    cpgenie.run_experiments(methylations, sequences_onehot, num_to_chr_dic, args.train_size, memory_chunk_size=args.memory_chunk_size)
