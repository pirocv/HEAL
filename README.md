# HEAL (Hierarchical Estimate from Agnostic Learning)
- Machine learning-based genome analysis and risk prediction framework.

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

### Usage
Run the HEAL script from the command line with the following arguments:

```
python HEAL.py --file_path <path_to_input_file> [options]
```

### Command-line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--file_path` | str | Yes | - | Full path to the input file |
| `--output` | str | No | Current working directory | Output path |
| `--splits` | int | No | 5 | Number of splits for cross-validation |
| `--trials` | int | No | 1 | Number of trials to run |
| `--l1` | float | No | 1.0 | Lower bound of lambda candidates |
| `--l2` | float | No | 40.0 | Upper bound of lambda candidates |
| `--lfidelity` | int | No | 5 | Fidelity of linspace of lambda candidates |
| `--scoring` | str | No | 'roc_auc' | Scoring metric to maximize |
| `--random_state` | int | No | 42 | Random state to start from |
| `--tts` | bool | No | False | Use train_test_split instead of StratifiedKFold for outer CV |

## Citation
Please cite the following paper
- Hirotaka Ieki, Kaoru Ito, Sai Zhang, Satoshi Koyama, Martin Kjellberg, Hiroki Yoshida, Ryo Kurosawa, Hiroshi Matsunaga, Kazuo Miyazawa, Nobuyuki Enzan, Changhoon Kim, Jeong-Sun Seo, Koichiro Higasa, Kouichi Ozaki, Yoshihiro Onouchi, Koichi Matsuda, Yoichiro Kamatani, Chikashi Terao, Fumihiko Matsuda, Michael Snyder, Issei Komuro "Machine Learning Reveals the Contribution of Rare Genetic Variants and Enhances Risk Prediction for Coronary Artery Disease in the Japanese Population" medRxiv 2024 doi.org/10.1101/2024.08.13.24311909

<img width="200" alt="snyderlab_logo" src="https://github.com/pirocv/HEAL/assets/51925146/0c17a201-9642-4da3-9457-5ff83ddb9a1b">
<img width="100" alt="hp_riken" src="https://github.com/pirocv/HEAL/assets/51925146/b37c836b-1a0e-4a2b-aca5-39baf220e4ea">
<img width="100" alt="cgi_logo" src="https://github.com/pirocv/HEAL/assets/51925146/d487a395-6741-4093-b515-ea078c685333">

