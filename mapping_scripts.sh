FastQC /path_to/reads_reverse.fastq.gz
java -jar trimmomatic-0.35.jar PE -phred33 input_forward.fq.gz input_reverse.fq.gz output_forward_paired.fq.gz output_forward_unpaired.fq.gz output_reverse_paired.fq.gz output_reverse_unpaired.fq.gz LEADING:5 TRAILING:10


bismark_genome_preparation ~/path_to_genome_folder/
bismark --genome ~/path_to_genome_folder/ -1 /path_to/reads_forward.fastq.gz -2 /path_to/reads_reverse.fastq.gz -o /output/ --multicore 10
bismark_methylation_extractor --gzip --bedGraph --CX --cytosine_report --genome_folder ~/path_to_genome_folder/ ~/path_to/mapping_output.bam -o ~/output/ --multicore 10
