import argparse
import os
import preprocess.data_reader as data_reader
import numpy as np
import preprocess.gene_body_methylation as gbm

if not os.path.exists('./dump_files/'):
    os.makedirs('./dump_files/')
if not os.path.exists('./models/'):
    os.makedirs('./models/')
parser = argparse.ArgumentParser(description='AMPS')

parser.add_argument('-m', '--methylation_file', help='methylation file address', required=True)
parser.add_argument('-g', '--genome_assembly_file', help='genome sequence file address, must be in fasta format', required=True)
parser.add_argument('-c', '--context', help='context', required=True)
parser.add_argument('-a', '--annotation_file', help='annotation file address', required=True)
parser.add_argument('-ct', '--coverage_threshold', help='minimum number of reads for including a cytosine in the training dataset', required=False, default=10, type=int)

args = parser.parse_args()

organism_name = args.organism_name
sequences = data_reader.readfasta(args.genome_assembly_file)
print('genome sequence assembly is loaded...')
methylations = data_reader.read_methylations(args.methylation_file,  args.coverage_threshold, context=args.context)
annot_df = data_reader.read_annot(args.annotation_file)
genes_df = data_reader.subset_annot(annot_df, 'gene')
usls_chrs = list(sequences.keys() - set(genes_df['chr'].unique()))
if len(usls_chrs) > 0:
    methylations = methylations[methylations.chr.isin(usls_chrs)==False]
    for val in usls_chrs:
        del sequences[val]
meth_seq = data_reader.make_meth_string(organism_name, methylations, sequences, args.coverage_threshold, from_file=True)
cntx_seq = data_reader.make_context_string(organism_name, methylations, sequences, from_file=True)

genes_avg_p, genes_avg_n, flac_up_avg_p, flac_up_avg_n, flac_down_avg_p, flac_down_avg_n = gbm.get_gene_meth(meth_seq, genes_df,  5, threshold=0.5, context=args.context, context_seq=cntx_seq)
final_p = np.concatenate((flac_down_avg_p, genes_avg_p, flac_up_avg_p))
final_n = np.concatenate((flac_down_avg_n, genes_avg_n, flac_up_avg_n))

np.save('./output/'+organism_name+'_genemethylation_nontemplate.npy', final_p)
np.save('./output/'+organism_name+'_genemethylation_nontemplate.npy', final_n)

print('output is saved in the: '+'./output/'+organism_name+'_genemethylation_nontemplate.npy' + 'and ' + './output/'+organism_name+'_genemethylation_nontemplate.npy')
