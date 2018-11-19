#!/usr/bin/env python
#coding:utf-8
import re
import random

import numpy as np
import matplotlib.pyplot as plt
from bayes import NaiveBayesClassifier

ENCODING='ISO-8859-1'
TRAIN_PERCENTAGE=0.9

def get_doc_vector(words, vocabulary):#one-hot vector
    doc_vec = [0]*len(vocabulary)
    
    for word in words:
        if word in vocabulary:
            idx = vocabulary.index(word)
            doc_vec[idx] = 1
    return doc_vec

def parse_line(line):
    label = line.split(',')[-1].strip()
    content = ','.join(line.split(",")[:-1])
    word_vec = [word.lower() for word in re.split(r'\W+', content) if word]
    return word_vec, label

def parse_file(filename):
    vocabulary, word_vecs, labels = [], [], []
    with open(filename, 'r', encoding=ENCODING) as f:
        for line in f:
            if line:
                word_vec, label = parse_line(line)
                vocabulary.extend(word_vec)
                word_vecs.append(word_vec)
                labels.append(label)
    vocabulary = list(set(vocabulary))
    return vocabulary, word_vecs, labels

if __name__ == "__main__":
    clf = NaiveBayesClassifier()
    vocabulary, word_vecs, labels = parse_file('english_big.txt')
    
    #训练数据&测试数据,这里改成先shuffle更好
    ntest = int(len(labels)*(1-TRAIN_PERCENTAGE))

    test_word_vecs = []
    test_labels = []
    for i in range(ntest):
        idx = random.randint(0, len(word_vecs)-1)
        test_word_vecs.append(word_vecs.pop(idx))
        test_labels.append(labels.pop(idx))
    
    train_word_vecs = word_vecs
    train_labels = labels
    
    train_dataset = [get_doc_vector(words, vocabulary) for words in train_word_vecs]

    #训练贝叶斯模型
    cond_probs, labels_probs = clf.train(train_dataset, train_labels)

    #测试模型
    error = 0
    for test_word_vec, test_label in zip(test_word_vecs, test_labels):
        test_data = get_doc_vector(test_word_vec, vocabulary)
        pred_label = clf.classify(test_data, cond_probs, cls_probs)
        if test_label != pred_label:
            print("Predict:{} -- Actual:{}").format(pred_label, test_label)
            error += 1
    print("Error Rate:{}".format(error/len(test_labels)))

    #绘制不同类型的概率分布曲线
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for label, probs in cond_probs.items():
        ax.scatter(np.arange(0, len(probs)),
                probs*label_probs[label],
                label=label,
                alpha=0.3)
        ax.legend()
    plt.show()
