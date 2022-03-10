import pandas as pd
import argparse
import preprocess.data_reader as dr

parser = argparse.ArgumentParser(description='AMPS')

parser.add_argument('-m', '--methylation_file', help='methylation file address', required=True)
parser.add_argument('-s', '-size', help='size of sample', required=True, type=int)
parser.add_argument('-o', '--output_address', help='output address', required=True)


def methylation_sampler(methylation_address, size, address):
     methylations = dr.read_methylations(methylation_address)
     methylations.sample(n=size, random_state=1).to_csv(address, header=None, index=None, sep=' ', mode='a')

args = parser.parse_args()
methylation_sampler(args.methylation_file, args.size, args.output_address)
print('sampled file is saved in: '+args.output_address)
