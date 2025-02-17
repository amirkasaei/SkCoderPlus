# SkCoder: A Sketch-based Approach for Automatic Code Generation

Official implementation of our ICSE 2023 paper on Automatic Code Generation. [Paper](https://arxiv.org/abs/2302.06144)

## Table of Contents

- Requirements
- Datasets
- Usage
- Releasing a trained model
- Acknowledgement
- Citation

## Requirements

- Java 1.8.0
- python 3.10
- pytorch 1.13.1
- transformers 4.24.0
- tqdm 4.60.0
- tree_sitter 0.2.0
- fire 0.5.0
- nltk
- tensorboard

## Datasets

In this paper, we conduct experiments on three datasets, including `HearthStone`, `Magic`, and `AixBench-L`. The raw datasets are available at [Google Drive](https://drive.google.com/drive/folders/1p04arpnmGT_QdeG_v5I-OkGj517wHXGI?usp=drive_link).

Please download the datasets and put them in the `data` folder. Taking the `HearthStone` dataset, the folder structure should be like this:

```
data
├── hearthstone
│   ├── train.jsonl
│   ├── dev.jsonl
│   ├── test.jsonl
│   ├── train_with_example.jsonl
│   ├── dev_with_example.jsonl
│   ├── test_with_example.jsonl
```

Each line of these `jsonl` files is a json object, which contains the following fields:

- input: str, the original input
- input_tokens[list]: list[str], the tokenized input
- output: str, the original output
- output_tokens[list]: list[str], the tokenized output


## Usage

### Step 1: Runing the Retriever

The retriever is used to retrieve the most similar code snippets from the code corpus. We have replaced the papers default retriver with the [CodeT5 embedding model](https://huggingface.co/Salesforce/codet5p-110m-embedding). This change improves retrieval accuracy and helps better capture semantic relationships between input queries and code snippets.


We run the retriever on the `HearthStone` dataset.
First, we extract input requirements from datasets and save them as files (`train.in`, `dev.in`, `test.in`).

```Bash
python process4retriever.py \
    --type preprocess \
    --data_path data/hearthstone \
```

Then, we utilze a search engine to retrieve similar code snippets from the train data.

```Bash
cd retriever
bash compile.sh
bash buildIndex.sh
bash buildExemplars.sh
```

Next, we extract similar code snippets and save them as `jsonl` files into `data/hearthstone`, including `{train, dev, test}_with_example.jsonl`.
Each file contains the following keys:

- input: str, the original input
- input_tokens[list]: list[str], the tokenized input
- output: str, the original output
- output_tokens[list]: list[str], the tokenized output
- examples: list[str], the Top-K similar code snippets

### Step 2: Training the Sketcher

After replacing the retriever with CodeT5, we further trained the model using the improved retrieval outputs. This helps refine the generated sketches and improves overall model performance.

As a disclaimer, the term `gcb` appearing everywhere stands for GraphCodeBert, which we use as a base model for fine-tuning.
It is here because there was a version that trains the model from scratch, so the term gcb is thus used to distinguish them.

#### Data Preprocessing

Run preprocess.py on the `{train,dev,test}_with_example.jsonl` to produce `{train,valid,test}_sketcher.json`.

```Bash
cd sketcher
python preprocess.py --input ../data/hearthstone/train_with_example.jsonl --output ../data/hearthstone/train_sketcher.json
python preprocess.py --input ../data/hearthstone/dev_with_example.jsonl --output ../data/hearthstone/dev_sketcher.json
python preprocess.py --input ../data/hearthstone/test_with_example.jsonl --output ../data/hearthstone/test_sketcher.json
```

#### Training
Run `run-hearthstone-gcb.sh`.
```Bash
export CUDA_VISIBLE_DEVICES=0 # the GPU(s) you want to use to train
bash run-hearthstone-gcb.sh test1
```
`test1` is the default `runs` folder used for the run. If you changed the folder name, you should change corresponding path in `eval.sh`.

#### Generating data for the editor
First run the inference script with the GPU.
```bash
export CUDA_VISIBLE_DEVICES=0 # the GPU(s) you want to use to inference
bash eval-hearthstone-gcb.sh
```
Then run `add_sketch.py` with the data folder path as the parameter to generate `{train,dev,test}_with_sketch.jsonl`.
```Bash
python add_sketch.py --data ../data/hearthstone
```

Finally, the sketcher outputs `{train,dev,test}_with_sketch.jsonl` in `data/hearthstone`. The format of these files is the same as the input of the retriever, except two more columns:

- sketch: a list of sketch from each example, where each sketch is a string. Note that we use <pad> instead of [PAD] .
- oracle-sketch: a list of oracle sketches for each example. The format is the same as sketch.

### Step 3: Training the Editor

The editor is train to generate code based on the requirement and code sketch. We run the editor on the `HearthStone` dataset.

#### Data Preprocessing
We run `process2editor.py` to generate the training data for the editor.

```Bash
python process4editor.py --data_path data/hearthstone
```

The generated data is saved in `data/hearthstone/{train,dev,test}_editor.jsonl`.

#### Training and Inference
Please modify the `ROOT_DIR` in `train.sh` and `inference.sh`, which denote the absolute path of the project.
```Bash
cd editor/sh
python run_exp.py --do_train --task hearthstone --gpu {gpu_ids}
```
Where `gpu_ids` is the GPU(s) you want to use to train, such as `0,1`.

`run_exp.py` will automatically train the model and generate the code for the test data. The generated code is saved in `editor/sh/saved_models/hearthstone/prediction/test_best-bleu.jsonl`. 

### Step 4: Evaluation
We evaluate the generated code using three metrics, including Exact Match (EM), BLEU, and CodeBLEU. We run the evaluation on the `HearthStone` dataset.

```Bash
cd evaluator
python evaluate.py --input_file {prediction_path} --lang {lang}
```

Where `prediction_path` is the path of the generated code, such as `../editor/sh/saved_models/hearthstone/prediction/test_best-bleu.jsonl`.
`lang` is the programming language of the generated code (`Hearthstone`: `python`, `Magic` and `AixBench-L`: `java`).

### Improvements with CodeT5 Embeddings
By replacing the original retriever with CodeT5 embeddings, we achieved the following improvements:
- More accurate retrieval of relevant code snippets, leading to better initial sketches
- Enhanced training quality by fine-tuning on the improved retrieval outputs
- Better performance across `Hearthstone` datasets in terms of BLEU and CodeBLEU scores

## Releasing a trained model
To facilitate the research community, we release the trained checkpoints of the sketcher and editor. The models are available at Google Drive([Sketcher](https://drive.google.com/drive/folders/1Vo48FC-pfX3FJwJihef6w2KWW-vjnu9B?usp=drive_link), [Editor](https://drive.google.com/drive/folders/17irATOV2xvle7Dq20-qydQDtV49PaG3C?usp=drive_link)). Please download the models and put them in the corresponding folders. Take the HearthStone dataset, the folder structure is as follows:

```
sketcher
├── runs
│   ├── gcb-hs
```


```
editor
├── sh
│   ├── saved_models
│   │   ├── hearthstone
```

Then, you can run the inference script to generate the code for the test data.

```Bash
cd editor/sh
python run_exp.py --task hearthstone --gpu {gpu_ids}
```


## Acknowledgement
The code is based on [Re2Com](https://github.com/Gompyn/re2com-opensource), [GraphCodeBERT](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT), and [CodeT5](https://github.com/salesforce/CodeT5). We thank the authors for their great work.

## Citation
If you find this repository useful, please cite our paper:
```
@inproceedings{SkCoder,
  author       = {Jia Li and
                  Yongmin Li and
                  Ge Li and
                  Zhi Jin and
                  Yiyang Hao and
                  Xing Hu},
  title        = {SkCoder: {A} Sketch-based Approach for Automatic Code Generation},
  booktitle    = {45th {IEEE/ACM} International Conference on Software Engineering,
                  {ICSE} 2023, Melbourne, Australia, May 14-20, 2023},
  pages        = {2124--2135},
  publisher    = {{IEEE}},
  year         = {2023},
  url          = {https://doi.org/10.1109/ICSE48619.2023.00179},
  doi          = {10.1109/ICSE48619.2023.00179},
  timestamp    = {Wed, 19 Jul 2023 10:09:12 +0200},
  biburl       = {https://dblp.org/rec/conf/icse/LiLLJHH23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
