import pandas as pd



def methylation_sampler(methylations, size ,address):
     methylations.sample(n=size, random_state=1).to_csv(address, header=None, index=None, sep=' ', mode='a')
