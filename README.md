comments by amber: :exclamation:
# RocketQA End-to-End QA-system Development Tool

This repository provides a simple and efficient toolkit for running RocketQA models and build a Question Answering (QA) system. 

## RocketQA
**RocketQA** is a series of dense retrieval models for Open-Domain QA. 

Open-Domain QA aims to find the answers of natural language questions from a large collection of documents. Common approaches often contain two stages, firstly a dense retriever selects a few relevant contexts, and then a neural reader extracts the answer.

RocketQA focuses on improving the dense contexts retrieval stage, and propose the following methods:
#### 1. [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2010.08191.pdf)

#### 2. [PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval](https://aclanthology.org/2021.findings-acl.191.pdf)

#### 3. [RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking](https://arxiv.org/pdf/2110.07367.pdf)


## Features
* ***State-of-the-art***, RocketQA models achieve SOTA performance on MSMARCO passage ranking dataset and Natural Question dataset.
* ***First-Chinese-model***, RocketQA-zh is the first open source Chinese dense retrieval model.
* ***Easy-to-use***, both python installation package and DOCKER environment are provided.
* ***Solution-for-QA-system***, developers can build an End-to-End QA system with one line of code.
  
  

## Installation

### Install package
First, install [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html).
```bash
# GPU version:
$ pip install paddlepaddle-gpu

# CPU version:
$ pip install paddlepaddle
```

Second, install rocketqa package:
```bash
$ pip install rocketqa
```

NOTE: RocketQA package MUST be running on Python3.6+ with [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html) 2.0+ :

### Download Docker environment

```bash
docker pull rocketqa_docker_name

docker run -it rocketqa_docker_name
```
  
[:exclamation:]
## API
The RocketQA development tool supports two types of models, ERNIE-based dual-encoder for answer retrieval and ERNIE-based cross-encoder for answer re-ranking. And the development tool provides the following methods:

#### `rocketqa.available_models()`

Returns the names of all available RocketQA models. 

[:exclamation:] 需要对这些模型在readme里进行简单的介绍吗（比如一个list来说明或者一个表格），受限于我的背景知识，我看了一会儿才看懂每个名字代表的是什么意思
```
dict_keys(['v1_marco_de', 'v1_marco_ce', 'v1_nq_de', 'v1_nq_ce', 'pair_marco_de', 'pair_nq_de', 'v2_marco_de', 'v2_marco_ce', 'v2_nq_de', 'zh_dureader_de', 'zh_dureader_ce'])
```
[:exclamation:] 以下function需要直接链接到对应的代码吗
#### `rocketqa.load_model(model, use_cuda=False, device_id=0, batch_size=1)`

Returns the model specified by the input parameter `model`. The RocketQA models returned by "available_models()" or your own checkpoint specified by a path can be initialized; both dual encoder and cross encoder can be initialized. 

---
[:exclamation:] 自己的checkpoint可以用这些api吗，还是只有Dual-encoder returned by "load_model()" 可以用这些api？建议描述清楚
Dual-encoder returned by "load_model()" supports the following methods:

#### `model.encode_query(query: List[str])`

Given a list of queries, returns their representation vectors encoded by model.

#### `model.encode_para(para: List[str], )`
[:exclamation:] model.encode_para(para: List[str], title: List[str] (optional)) 把title是什么输入格式也写进去会不会更好
Given a list of passages (their corresponding titles are optional), returns their representations vectors encoded by model.

#### `model.matching(query: List[str], para: List[str], )`
[:exclamation:] model.matching(query: List[str], para: List[str], title: List[str] (optional)) 把title是什么输入格式也写进去会不会更好
[:exclamation:] 这个function的代码实现是para是一个list的qp对，如果我向para只输入p的list会报错。是代码错了？还是在这里写清楚para是一个list的qp对？
Given a list of queries and a list of their corresponding passages (their corresponding titles are optional), returns their matching scores (dot product between two representation vectors). 

---
[:exclamation:] 自己的checkpoint可以用这些api吗，还是只有Cross-encoder returned by "load_model()" 可以用这些api？建议描述清楚
Cross-encoder returned by "load_model()" supports the following method:

#### `model.matching(query: List[str], para: List[str], )`
[:exclamation:] model.matching(query: List[str], para: List[str], title: List[str] (optional)) 把title是什么输入格式也写进去会不会更好
[:exclamation:] 这个function的代码逻辑是para是一个list的qp对，如果我向para只输入一个list的p就会报错。是代码逻辑错了？还是在这里写清楚para是一个list的qp对？
Given a list of queries and a list of their corresponding passages (their corresponding titles are optional), returns their matching scores (probability that the paragraph is the query's right answer).
  
  

## Examples

A short example about how to use RocketQA. 

###  Run RocketQA Model
To run RocketQA models, developers should set the parameter `model` in 'load_model()' method with RocketQA model name return by 'available_models()' method. 
[:exclamation:] inner_products = dual_encoder.matching(query=query_list, para=para_list) 这个function 用以下例子输入会报错，错误还是像我上一个说的，代码写错了
[:exclamation:] 建议用dot product代替inner product
```python
import rocketqa

query_list = ["trigeminal definition"]
para_list = ["Definition of TRIGEMINAL. : of or relating to the trigeminal nerve.ADVERTISEMENT. of or relating to the trigeminal nerve. ADVERTISEMENT."]

# init dual encoder
dual_encoder = rocketqa.load_model(model="v1_marco_de", use_cuda=True, batch_size=16)

# encode query & para
q_embs = dual_encoder.encode_query(query=query_list)
p_embs = dual_encoder.encode_para(para=para_list)
# compute inner product of query representation and para representation
inner_products = dual_encoder.matching(query=query_list, para=para_list)
```

### Run Your Own Model
[:exclamation:] 改了这章的标题，我没有自己的model所以没有测试这一块的代码，但是matching这个function估计有相同的问题，建议检查
To run checkpoints, developers should write a config file, and set the parameter `model` in 'load_model()' method with the path of the config file.

```python
import rocketqa

query_list = ["交叉验证的作用"]
title_list = ["交叉验证的介绍"]
para_list = ["交叉验证(Cross-validation)主要用于建模应用中，例如PCR 、PLS回归建模中。在给定的建模样本中，拿出大部分样本进行建模型，留小部分样本用刚建立的模型进行预报，并求这小部分样本的预报误差，记录它们的平方加和。"]

# conf
ce_conf = {
    "model": "./own_model/config.json",     # path of config file
    "use_cuda": True,
    "device_id": 0,
    "batch_size": 16
}

# init cross encoder
cross_encoder = rocketqa.load_model(**ce_conf)

# compute matching score of query and para
ranking_score = cross_encoder.matching(query=query_list, para=para_list, title=title_list)
```

The config file is a JSON format file.
```bash
{
    "model_type": "cross_encoder",
    "max_seq_len": 160,
    "model_conf_path": "en_large_config.json",  # path relative to config file
    "model_vocab_path": "en_vocab.txt",         # path relative to config file
    "model_checkpoint_path": "marco_cross_encoder_large", # path relative to config file
    "joint_training": 0
}
```
  


## Start your QA-System
[:exclamation:] 这部分描述上没有什么问题，受限于我的背景知识，我不是很理解index，search是什么意思，建议写完整这几句comments；我没有run这部分的代码，建议确认一下代码都能正常运行。
With the examples below, developers can build own QA-System

### Running with JINA
```bash
cd examples/jina_example/
pip3 install -r requirements.txt

# Index
python3 app.py index

# Search
python3 app.py query
```



### Running with Faiss

```bash
cd examples/faiss_example/
pip3 install -r requirements.txt

# Index
python3 index.py ${language} ${data_file} ${index_file}

# Start service
python3 rocketqa_service.py ${language} ${data_file} ${index_file}

# request
python3 query.py
```

