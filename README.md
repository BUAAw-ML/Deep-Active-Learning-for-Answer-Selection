# Deep-Active-Learning-for-Answer-Selection

## Abstract
Given a question and a set of candidate answers, answer selection is the task of identifying the best answer, which can be viewed as a kind of learning-to-rank tasks. Learning to rank arises in many information retrieval applications, where deep learning models can achieve inspiring results. Training a deep learning model often requires large scale annotated data that are expensive and time-consuming to obtain. Active learning presents a promising approach to this problem by selecting more informative training data to reduce the amount of labelling efforts required. Because traditional active learning methods cannot be directly used for deep learning, researchers have proposed multiple deep active learning methods. However, none of the previous research efforts on deep active learning algorithms presents a specific framework for learning-to-rank tasks. In this work, we introduce a novel deep active learning method DELO based on deep expected loss optimization for the answer selection task. It adopts a data acquisition function based on model uncertainty with Bayesian deep learning and the expected loss optimization. Moreover, a two-step batch-mode procedure, combining DELO and other data acquisition strategies is proposed to further improve the performance of active learning. 

## Publication
This is the codebase for our 2021 paper on TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING (TKDE):
[Deep Bayesian Active Learning for Learning to Rank: A Case Study in Answer Selection](https://ieeexplore.ieee.org/document/9347711).

```
@article{wang2021deep,
  title={Deep Bayesian Active Learning for Learning to Rank: A Case Study in Answer Selection},
  author={Wang, Qunbo and Wu, Wenjun and Qi, Yuxing and Zhao, Yongchi},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2021},
  publisher={IEEE}
}
```

## Requirements: 
- Python3;
- Pytorch;
- numpy;
- sklearn;

In addition, anyone who want to run these codes should download the word embedding 'glove.6B.300d.txt' from https://nlp.stanford.edu/projects/glove/. The file should be placed at './datasets/word_embedding/glove.6B.300d.txt'.

## Datasets:
We upload a subset of the dataset YahooCQA to run these codes. You can download other datasets of Community-based Question Answering (CQA) and place it at './datasets'.

## Train and Test
python main.py

- The options of the active learning method can be set in 'main.py'. 