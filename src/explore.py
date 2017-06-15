import numpy as np
import pandas as pd
import json
import os
import seaborn as sns
from collections import Counter

data_folder = '../../data/grml/grammarly_research_assignment/data'

sentence_train = 'sentence_train'
sentence_test = 'sentence_test'
sentence_private_test = 'sentence_private_test'

corrections_train = 'corrections_train'
corrections_test = 'corrections_test'

parse_train = 'parse_train'
parse_test = 'parse_test'
parse_private_test = 'parse_private_test'

pos_tags_train, pos_tags_test, pos_tags_private_test = \
    'pos_tags_train', 'pos_tags_test', 'pos_tags_private_test'


ngrams = 'ngrams'
idioms = os.path.join(data_folder, 'idioms')

def get(name):
    return os.path.join(data_folder, name)


def load_resource(name):
    with open(get(name+'.txt')) as f:
        return json.load(f)


def load_train():
    sent_train = load_resource(sentence_train)
    corr_train = load_resource(corrections_train)
    print len(sent_train)==len(corr_train)
    return [zip(sent_train[i], corr_train[i]) for i in range(len(sent_train))]


def to_line(l):
    return ' '.join(l)

def count_errors(l):
    return sum(len([y for y in x if y is not None]) for x in l)

def explore_errors_counts(arr):
    l=[]
    for x in arr:
        for y in x:
            if y[1] is not None:
                l.append((y[0], y[1]))

    return Counter(l)

def explore_errors_pairs(arr):
    l=[]
    blja = []
    for x in arr:
        for i in range(len(x)):
            y = x[i]
            if y[1] is not None:
                if i+1>=len(x):
                    blja.append(x)
                    continue
                y_next = x[i+1]
                l.append((y[0]+' '+y_next[0], y[1]+' '+y_next[0]))

    print len(blja)

    return Counter(l)