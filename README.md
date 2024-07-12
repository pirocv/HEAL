# HEAL
- Machine learning-based genome analysis and risk prediction model.

## supported file type
- csv file (Mutation burden matrix)
   - row: sample ID, column: gene name

## requirements
- python3
- pandas
- numpy
- scikit-learn
- scipy

## How to run
### Step0 Prepare mutation burden matrix.
1. Annotate VCF file of whole exome or genome sequencing data with gene name, deleteriousness score, and allele frequency info.
2. Preprosess annotated genotype data to calculate mutation burden. Sample mutation burden file is available in `toy_data`.
### Step 1 Run the HEAL script
1. Input file: Mutation burden matrix.
### Step 2 The model outputs
1. Disease gene lists.
2. Genetic risk prediction model.
3. Prediction performance summary.

## Citation
Please cite the following paper
- ......

<img width="200" alt="snyderlab_logo" src="https://github.com/pirocv/HEAL/assets/51925146/0c17a201-9642-4da3-9457-5ff83ddb9a1b">
<img width="100" alt="hp_riken" src="https://github.com/pirocv/HEAL/assets/51925146/b37c836b-1a0e-4a2b-aca5-39baf220e4ea">
<img width="100" alt="cgi_logo" src="https://github.com/pirocv/HEAL/assets/51925146/d487a395-6741-4093-b515-ea078c685333">

