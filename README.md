# AMPS

AMPS is a python tool for context-specific DNA methylation prediction using a deep neural network. The prediction can be based on the sequence, sequence+annotation or methylation level of neighboring Cytosines.

## Dependencies

1. Tensorflow 2.3.+
2. biopython
3. pandas
4. numpy

## Preprocess

### Mapping WGBS data
If you have the WGBS reads, you should first get to the mapping process. The file mapping_scripts.txt shows a sample for the steps:

- Check the quality of the reads
- Trim the reads with low quality
- Map the reads using Bismark:
  - Prepare genome
  - Map the reads
  - genome-wide methylation extractor

## Inputs

AMPS uses three inputs: 
+ DNA Sequence, which should be in Fasta format
+ Methylations, which contains the methylation and context information of cytosines. This file is based on the output of methylation extractor of [Bismark](https://github.com/FelixKrueger/Bismark/tree/master/Docs#optional-genome-wide-cytosine-report-output "wgbs mapping tool") <p><code>chromosome position strand count-methylated count-unmethylated C-context trinucleotide-context</code></p>
+ annotaions, Which is a table containing the annotated functiona element coordination and must be in GF3 format.


### Sequence based

<code>train.py</code> is the module for training the AMPS. AMPS will be trained for the methylation prediction from sequence and annotation. If the annotation file is not passed to the module, it will be trained based on the sequence only. Module options:

1. <code> -m, --methylation_file: methylation file address, required</code>
2. <code> -g, --genome_assembly_file: genome sequence file address, must be in fasta format, required</code>
3. <code> -c, --context: context, required</code>
4. <code> -a, --annotation_file: annotation file address</code>
5. <code> -ia, --include_annotation: does the predictor include the annotation in the input? True/False</code>
6. <code> -tr, --train_size: training dataset size, number of inputs for training</code>
7. <code> -ws, --window_size: window size, number of including nucleutides in a window</code>
8. <code> -ct, --coverage_threshold: minimum number of reads for including a cytosine in the training/testing dataset</code>
9. <code> -on, --organism_name: sample name, for saving the files</code>
10. <code> -mcs, --memory_chunk_size: number of inputs in each memory load</code>

As a sample you can run:

<code>python train.py -m ./sample/sample_methylations_train.txt -g ./sample/sample_seq.fasta -a ./sample/sample_annotation.txt -c CG</code>

This module will train a model and save it in the ./models/ directory. The saved model can be loaded and used for the desired set of cytosines. For using the model <code> test.py </code> should be used.

<code> test.py </code> loads the trained model to predict the binary methylation status of all the cytosines listed in the methylation file. The output of the prediction is a binary vector. Each vector element corresponds to a cytosine in the provided methylation file. This vector will be saved in the ./output/ folder. Module options:


1. <code> -m, --methylation_file: methylation file address, required</code>
2. <code> -mdl, --model_address: trained model address, required</code>
3. <code> -g, --genome_assembly_file: genome sequence file address, must be in fasta format, required</code>
4. <code> -a, --annotation_file: annotation file address</code>
5. <code> -ia, --include_annotation: does the predictor include the annotation in the input? True/False</code>
6. <code> -ws, --window_size: window size, number of including nucleutides in a window</code>
7. <code> -on, --organism_name: sample name, for saving the files</code>

As a sample you can run:

<code>python test.py -mdl ./models/sample_organismCG.mdl/ -m ./sample/sample_methylations_test.txt -g ./sample/sample_seq.fasta -a ./sample/sample_annotation.txt</code>

The output is a text file containing a binary vector saved in <code> ./output/ </code> folder.

For calculating the accuracy, you can use <code> accuracy_clc.py </code> module. It gets the predicted binary vectory and either a methylation file or another binary vector. If a methylation file is provided in the input, the module calculates the methylation status of each cytosine in the methylation file and then compare it with the predicted binary vector. Else two binary vectors are compared together. The accuracy is the number of correct prediction over all the test size.

Module options:

1. <code> -pr, --y_predicted: address to the predicted binary vector file, required</code>
2. <code> -te, --y_true: address to true methylation status binary vector file</code>
3. <code> -m, --methylation_file: address to true methylation file address</code>


sample code:

<code>python accuracy_clc.py -pr ./output/sample_organism.txt -m ./sample/sample_methylations_test.txt</code>

### Methylation-profile based

The <code> train_methprofile.py </code> traines a model for cytosine methylation prediction based on its neghbouring Cytosine methylation levels. Module options:

1. <code> -m, --methylation_file: methylation file address, required</code>
2. <code> -c, --context: context, required</code>
3. <code> -tr, --train_size: training dataset size, number of inputs for training</code>
4. <code> -ws, --window_size: window size, number of including nucleutides in a window</code>
5. <code> -ct, --coverage_threshold: minimum number of reads for including a cytosine in the training/testing dataset</code>
6. <code> -on, --organism_name: sample name, for saving the files</code>

The trained model will be saved on the ./models/ folder. Then by using the <code> test_methprofile.py </code> for a sample of cytosines, the binary methylation status can be predicted. This module's input is a profile of a set of cytosines provided in a tab-separated file. Each row of the file should contain the methylation levels of the neighboring cytosines. For example, below is a cytosine profile with a window size of 20 centered on the unknown cytosine(methylation levels of 10 cytosines downstream and ten cytosines upstream)

<code>0.76190476, 0.67857143, 0.6875    , 0.94366197, 1.        , 0.88235294, 0.6875    , 0.91304348, 0.94444444,1.        , 0.92      , 0.8125    , 0.91666667, 0.81481481, 0.82758621, 0.60606061, 0.95833333, 1.        , 1.        , 0.92307692
 </code>
This can be a row in the cytosine profiles file. The inputs of the <code> test_methprofile.py </code> module are:

1. <code> -p, --prfiles_address: address to the file containing the cytosine profiles. a tab separated file, each row is the methylation level of neighboring Cytosines, required</code>
2. <code> -mdl, --model_address: trained model address, required</code>
3. <code> -on, --organism_name: sample name, for saving the files</code>

The output is a text file containing a binary vector saved in <code> ./output/ </code> folder.

### GeneBody methylation

The methylation analysis in the gene-body and flanking regions are implemented in the <code> gene_body_analysis.py </code>. This module divides the flanking regions and gene body into several bins and then in the genome-wide calculates the average methylation in each bin. Module inputs:

1. <code> -m, --methylation_file: methylation file address, required</code>
2. <code> -g, --genome_assembly_file: genome sequence file address, must be in fasta format, required</code>
3. <code> -c, --context: context, required</code>
4. <code> -a, --annotation_file: annotation file address</code>
5. <code> -ct, --coverage_threshold: minimum number of reads for including a cytosine in the training/testing dataset</code>
6. <code> -bn, --bin_number: number of bins for genebody and flanking regions</code>


The output is two NumPy vectors, each containing the average methylation levels for the bins in the template or nontemplate strands. The numbers come in the order of downstream flanking region, gene body, and upstream flanking region.
