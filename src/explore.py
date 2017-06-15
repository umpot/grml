import numpy as np
import pandas as pd
import json
import os
import seaborn as sns
from collections import Counter, defaultdict

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

ARTICLES={'a', 'an', 'the'}


NOUNS={'NNPS', 'NNP', 'NNS', 'NN'}

def get(name):
    return os.path.join(data_folder, name)


def load_resource(name):
    with open(get(name+'.txt')) as f:
        return json.load(f)


def load_train():
    sent_train = load_resource(sentence_train)
    corr_train = load_resource(corrections_train)
    pos_train = load_resource(pos_tags_train)
    print len(sent_train)==len(corr_train)
    res= [zip(sent_train[i], corr_train[i], pos_train[i]) for i in range(len(sent_train))]
    res =[[list(y) for y in x] for x in res]

    for x in res:
        build_noun_chunks(x)

    return res


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

def explore_pairs_freq(arr):
    res=defaultdict(list)
    for x in arr:
        for i in range(len(x)):
            y = x[i]
            art = y[0]
            if art not in ARTICLES:
                continue

            y_next = x[i+1]

            key =art+' '+ y_next[0]

            m = res[key]
            # if 'count' not in m:
            #     m['count']=0
            #
            # m['count']+=1


            m.append(y[1])

    res = [(k,v) for k,v in res.iteritems()]
    res.sort(key=lambda s: len(s[1]), reverse=True)

    return res


bad = 0
def build_noun_chunks(x):
    global bad
    for i, y in enumerate(x):
        # print y
        article = y[0]
        sub =y[1]
        pos =y[2]

        if article not in ARTICLES:
            continue

        if sub is None:
            sub =article

        original_chunk = [article]
        correct_chunk = [sub]

        noun = None

        while True:
            i+=1
            if i==len(x):
                print ' '.join(original_chunk)
                bad+=1
                noun = 'NONE'
                key = article+' '+noun
                y.append(key)
                y.append(original_chunk)
                y.append(correct_chunk)
                y.append(noun)
                break
            new_token = x[i][0]
            new_token_pos = x[i][2]
            original_chunk+=[new_token]
            correct_chunk+=[new_token]
            if new_token_pos in NOUNS:
                noun = new_token

                key = article+' '+noun
                y.append(key)
                y.append(original_chunk)
                y.append(correct_chunk)
                y.append(noun)
                break

    print 'bad {}'.format(bad)


def extract__articles_chunks(arr):
    l=[]
    for x in arr:
        for y in x:
            if len(y)>3:
                l.append(y)



    m = {
        'key':[x[3] for x in l],
        'original_chunk':[x[4] for x in l],
        'correct_chunk':[x[5] for x in l],
        'noun':[x[6] for x in l],
        'correct':[1 if x[4]==x[5] else 0 for x in l],

    }

    return pd.DataFrame(m)








