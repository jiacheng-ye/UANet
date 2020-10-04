## Introduction

Codes for our EMNLP 2020 paper **Uncertainty-Aware Label Refinement for Sequence Labeling**. 

## Requirement
```
Python: 3.6 or higher.
PyTorch 1.0 or higher.
```
## Setup
Download Glove embedding from [here](https://nlp.stanford.edu/projects/glove/).

## Input format:
We use standard CoNLL format with each character and its label split by a whitespace in a line. The "BMES" tag scheme is prefered. 
Make sure to use `-DOCSTART-` to indicate the begining of a document.

A example from CoNLL2003 (additional pos/chunk features are not used in our experiments):
```
-DOCSTART- -X- -X- O

EU NNP B-NP S-ORG
rejects VBZ B-VP O
German JJ B-NP S-MISC
call NN I-NP O
to TO B-VP O
boycott VB I-VP O
British JJ B-NP S-MISC
lamb NN I-NP O
. . O O

Peter NNP B-NP B-PER
Blackburn NNP I-NP E-PER
```

## Usage

### training
run:
```
CUDA_VISIBLE_DEVICES=0 python main.py --train_dir 'data/conll2003/train.txt' --dev_dir 'data/conll2003/dev.txt' --test_dir 'data/conll2003/test.txt'  --model_dir 'outs' --word_emb_dir 'data/glove.6B.100d.txt' \
--model2_layer 2 --n_head 7 --d_head 160 --threshold 0.45 --model2_dropout 0.15 --attention_dropout 0.05
```

### decoding
run:
```
CUDA_VISIBLE_DEVICES=0 python main.py --status decode --model_dir <model dir> --raw_dir <file to be predicted>
```

### models 
We upload a model trained on CoNLL2003 dataset [here](https://drive.google.com/drive/folders/1wwQWmfXaxS3GbrR7G-k1XI0gQbp3ZqzV?usp=sharing). 

## Cite
If you use our code, please cite our paper as follows:


```
@inproceedings{gui2020uncertainty,
author = {Tao Gui, Jiacheng Ye, Qi Zhang, Zhengyan Li, zichu fei, Yeyun Gong and Xuanjing Huang},
title = {Uncertainty-Aware Label Refinement for Sequence Labeling},
publisher = {EMNLP},
year = {2020}
}
```

## Reference
- [NCRF++](https://github.com/jiesutd/NCRFpp.git)  