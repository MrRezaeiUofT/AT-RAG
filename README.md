# advanced_qa_graph


## Datasets
* You can download multi-hop datasets (MuSiQue, HotpotQA, and 2WikiMultiHopQA) from https://github.com/StonyBrookNLP/ircot.
```bash
# Download the preprocessed datasets for the test set.
$ bash ./download/processed_data.sh
# Prepare the dev set, which will be used for training our query complexity classfier.
$ bash ./download/raw_data.sh
$ python processing_scripts/subsample_dataset_and_remap_paras.py musique dev_diff_size 500
$ python processing_scripts/subsample_dataset_and_remap_paras.py hotpotqa dev_diff_size 500
$ python processing_scripts/subsample_dataset_and_remap_paras.py 2wikimultihopqa dev_diff_size 500

# Build index
python retriever_server/build_index.py {dataset_name} # hotpotqa, 2wikimultihopqa, musique
```

* You can download single-hop datasets (Natural Question, TriviaQA, and SQuAD) from https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py.
```bash
# Download Natural Question
$ mkdir -p raw_data/nq
$ cd raw_data/nq
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz
$ gzip -d biencoder-nq-dev.json.gz
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz
$ gzip -d biencoder-nq-train.json.gz

# Download TriviaQA
$ cd ..
$ mkdir -p trivia
$ cd trivia
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz
$ gzip -d biencoder-trivia-dev.json.gz
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz
$ gzip -d biencoder-trivia-train.json.gz

# Download SQuAD
$ cd ..
$ mkdir -p squad
$ cd squad
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-dev.json.gz
$ gzip -d biencoder-squad1-dev.json.gz
$ wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-train.json.gz
$ gzip -d biencoder-squad1-train.json.gz

# Download Wiki passages. For the singe-hop datasets, we use the Wikipedia as the document corpus.
$ cd ..
$ mkdir -p wiki
$ cd wiki
$ wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
$ gzip -d psgs_w100.tsv.gz

# Process raw data files in a single standard format
$ python ./processing_scripts/process_nq.py
$ python ./processing_scripts/process_trivia.py
$ python ./processing_scripts/process_squad.py

# Subsample the processed datasets
$ python processing_scripts/subsample_dataset_and_remap_paras.py {dataset_name} test 500 # nq, trivia, squad
$ python processing_scripts/subsample_dataset_and_remap_paras.py {dataset_name} dev_diff_size 500 # nq, trivia, squad

# Build index 
$ python retriever_server/build_index.py wiki
```

You can ensure that dev and test sets do not overlap, with the code below.
```bash
$ python processing_scripts/check_duplicate.py {dataset_name} # nq, trivia, squad hotpotqa, 2wikimultihopqa, musique
```
We provide the preprocessed datasets in [`processed_data.tar.gz`](./processed_data.tar.gz).