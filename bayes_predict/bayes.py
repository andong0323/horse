#!/usr/bin/env python
#coding:utf-8
from collections import defaultdict

import numpy as np

class NaiveBayesClassifier(object):
    def train(self, dataset, labels):
        sub_datasets = defaultdict(lambda: [])
        label_cnt = defaultdict(lambda: 0)

        for doc_vec, label in zip(dataset, labels):
            sub_datasets[label].append(doc_vec)
            label_cnt[label] += 1
        
        #计算类型概率
        label_probs = {k:v/len(labels) for k,v in label_cnt.items()}

        #计算不同类型的条件概率
        cond_probs = {}
        dataset = np.array(dataset)
        for label, sub_dataset in sub_datasets.items():
            sub_dataset = np.array(sub_dataset)
            #improve the classifier
            cond_prob_vec = np.log((np.sum(sub_dataset, axis=0) + 1) / (np.sum(dataset) + 2))
            cond_probs[label] = cond_prob_vec
        return cond_probs, label_probs

    def classify(self, doc_vec, cond_probs, label_probs):
        pred_probs = {}
        for label, label_prob in label_probs.items():
            cond_prob_vec = cond_probs[label]
            pred_probs[label] = np.sum(cond_prob_vec*doc_vec) + np.log(label_prob)
        return max(pred_probs, key=pred_probs.get)

