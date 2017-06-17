# coding=utf-8
import numpy as np
import pandas as pd
import json
import os
import seaborn as sns
from collections import Counter, defaultdict, OrderedDict
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
import json
import re

sns.set(color_codes=True)
sns.set(style="whitegrid", color_codes=True)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 100)

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
syntactic_ngrams = 'syntactic_ngrams'
idioms = 'idioms'
transcriptions = 'transcriptions'

singular_nouns = 'singular_nouns'
plural_nouns = 'plural_nouns'
uncountable_nouns = 'uncountable_nouns'

ARTICLES = {'a', 'an', 'the'}

NOUNS = {'NNPS', 'NNP', 'NNS', 'NN'}

VOWELS = {
    u'\u0259',  # 'ə',
    u'\u025b',  # 'ɛ',
    u'\u025a',  # 'ɚ',
    u'\u025d',  # 'ɝ',
    u'\u025c',  # 'ɜ',
    u'\u01d0',  # 'ǐ',
    u'\u0251',  # 'ɑ',
    u'\u0252',  # 'ɒ',
    u'\u0254',  # 'ɔ',
    u'\xe6',  # 'æ',
    u'\u0268',  # 'ɨ',
    u'\u026a',  # 'ɪ',
    u'\u026f',  # 'ɯ',
    u'\u1d7b',  # 'ᵻ',
    'A',
    'E',
    'I',
    'O',
    u'\u028c',  # 'ʌ',
    u'\u028a',  # 'ʊ',
    'a',
    'e',
    'i',
    'o',
    'u',
    u'\u01ce',
    u'\u0259'

}

symbols_to_skip = '[\'\(\-\{\@]'  # {"'", '(', '-', '{'}


def get(name):
    return os.path.join(data_folder, name)


def load_resource(name):
    with open(get(name + '.txt')) as f:
        return json.load(f, encoding='utf-8')


def load_transcription():
    return load_resource(transcriptions)


def build_starts_with_vowel_dict():
    trans = load_transcription()
    del trans['amens']
    trans = {k: re.sub(symbols_to_skip, '', v) for k, v in trans.iteritems()}

    print [(k, v) for k, v in trans.iteritems() if len(v) == 0]

    return {k.lower(): 1 if v[0] in VOWELS else 0 for k, v in trans.iteritems()}


def build_tri_starts_with_vowel_dict(starts_with_vowel_dict):
    res = {k: v for k, v in starts_with_vowel_dict.iteritems() if len(k) > 2}
    res = {k[:2]: v for k, v in res.iteritems()}

    return res


def build_bi_starts_with_vowel_dict(starts_with_vowel_dict):
    res = {k: v for k, v in starts_with_vowel_dict.iteritems() if len(k) > 1}
    res = {k[:1]: v for k, v in res.iteritems()}

    return res


def build_one_starts_with_vowel_dict():
    res = {chr(i): 0 for i in range(97, 123)}
    res['a'] = 1
    res['i'] = 1
    res['u'] = 1
    res['o'] = 1
    res['e'] = 1

    return res


starts_with_vowel_dict = build_starts_with_vowel_dict()
tri_starts_with_vowel_dict = build_tri_starts_with_vowel_dict(starts_with_vowel_dict)
bi_starts_with_vowel_dict = build_bi_starts_with_vowel_dict(starts_with_vowel_dict)

one_starts_with_vowel_dict = build_one_starts_with_vowel_dict()


def load_train():
    sent_train = load_resource(sentence_train)
    corr_train = load_resource(corrections_train)
    pos_train = load_resource(pos_tags_train)
    print len(sent_train) == len(corr_train)
    res = [zip(sent_train[i], corr_train[i], pos_train[i]) for i in range(len(sent_train))]
    res = [[list(y) for y in x] for x in res]

    for x in res:
        build_noun_chunks(x)

    return res


def load_ngrams():
    return load_resource(ngrams)


def load_syntactic_ngrams():
    return load_resource(syntactic_ngrams)


def to_line(l):
    return ' '.join(l)


def count_errors(l):
    return sum(len([y for y in x if y is not None]) for x in l)


def explore_errors_counts(arr):
    l = []
    for x in arr:
        for y in x:
            if y[1] is not None:
                l.append((y[0], y[1]))

    return Counter(l)


def explore_errors_pairs(arr):
    l = []
    blja = []
    for x in arr:
        for i in range(len(x)):
            y = x[i]
            if y[1] is not None:
                if i + 1 >= len(x):
                    blja.append(x)
                    continue
                y_next = x[i + 1]
                l.append((y[0] + ' ' + y_next[0], y[1] + ' ' + y_next[0]))

    print len(blja)

    return Counter(l)


def explore_pairs_freq(arr):
    res = defaultdict(list)
    for x in arr:
        for i in range(len(x)):
            y = x[i]
            art = y[0]
            if art not in ARTICLES:
                continue

            y_next = x[i + 1]

            key = art + ' ' + y_next[0]

            m = res[key]
            # if 'count' not in m:
            #     m['count']=0
            #
            # m['count']+=1


            m.append(y[1])

    res = [(k, v) for k, v in res.iteritems()]
    res.sort(key=lambda s: len(s[1]), reverse=True)

    return res


bad = 0


# def process_arr(x):
#


def build_noun_chunks(x):
    global bad
    for i, y in enumerate(x):
        # print y
        article = y[0]
        sub = y[1]
        pos = y[2]

        if article not in ARTICLES:
            continue

        if sub is None:
            sub = article

        original_chunk = [article]
        correct_chunk = [sub]

        noun = None
        next_token = None
        while True:
            i += 1
            if i == len(x):
                if next_token is None and new_token not in {'"', "'", ','}:
                    next_token = new_token
                print ' '.join(original_chunk)
                bad += 1
                noun = 'NONE'
                key = article + ' ' + noun
                y.append(key)
                y.append(original_chunk)
                y.append(correct_chunk)
                y.append(noun)
                y.append(next_token)
                break
            new_token = x[i][0]
            if next_token is None and new_token not in {'"', "'"}:
                next_token = new_token
            new_token_pos = x[i][2]
            original_chunk += [new_token]
            correct_chunk += [new_token]
            if new_token_pos in NOUNS:
                noun = new_token

                key = article + ' ' + noun
                y.append(key)
                y.append(original_chunk)
                y.append(correct_chunk)
                y.append(noun)
                y.append(next_token)
                break

    print 'bad {}'.format(bad)


key = 'key'
original_chunk = 'original_chunk'
correct_chunk = 'correct_chunk'
definite_chunk = 'definite_chunk'
indefinite_chunk = 'indefinite_chunk'

noun = 'noun'
correct = 'correct'
def_correct = 'def_correct'
sentence = 'sentence'
next_token = 'next_token'

st_with_v = 'st_with_v'

def_ngram_count = 'def_ngram_count'
indef_ngram_count = 'indef_ngram_count'


def create_df(arr):
    l = []
    for x in arr:
        for y in x:
            if len(y) > 3:
                l.append((y, x))

    m = {
        'key': [x[0][3] for x in l],
        'original_chunk': [x[0][4] for x in l],
        'correct_chunk': [x[0][5] for x in l],
        'noun': [x[0][6] for x in l],
        'next_token': [x[0][7] for x in l],
        'correct': [1 if x[0][4] == x[0][5] else 0 for x in l],
        'sentence': [x[1] for x in l]
    }

    df = pd.DataFrame(m)

    df[sentence] = df[sentence].apply(lambda s: ' '.join(x[0] for x in s))

    df[original_chunk] = df[original_chunk].apply(lambda s: ' '.join(s))
    df[correct_chunk] = df[correct_chunk].apply(lambda s: ' '.join(s))
    df[st_with_v] = df[next_token].apply(starts_with_vowel)
    df[key] = df[original_chunk].apply(lambda s: ' '.join(re.sub('[\(\"]', ' ', s).split()))

    def create_def_chunk(s):
        suff = ' '.join(s.split()[1:])
        return 'the ' + suff

    def create_indef_chunk(s, starts_with_vowel):
        suff = ' '.join(s.split()[1:])
        return 'an ' + suff if starts_with_vowel == 1 else 'a ' + suff

    def is_def_correct(s):
        return 1 if 'the'==s.split()[0] else 0

    df[def_correct] = df[correct_chunk].apply(is_def_correct)

    df[definite_chunk] = df[original_chunk].apply(create_def_chunk)
    df[indefinite_chunk] = df.apply(lambda s: create_indef_chunk(s[original_chunk], s[st_with_v]), axis=1)


    ngrams = load_ngrams()

    df[def_ngram_count] = df[definite_chunk].apply(lambda s: ngrams.get(s, 0))
    df[indef_ngram_count] = df[indefinite_chunk].apply(lambda s: ngrams.get(s, 0))

    cols = [def_ngram_count, indef_ngram_count, def_correct, original_chunk, correct_chunk, definite_chunk, correct,indefinite_chunk, noun, next_token,
            st_with_v, sentence]
    df = df[cols]

    return df


def get_count_mean_df(df):
    return df.groupby(key)[correct]. \
        agg({'count': 'count', 'mean': 'mean'}). \
        sort_values(by='count', ascending=False)


def starts_with_vowel(s):
    s = s.lower()
    if s in starts_with_vowel_dict:
        return starts_with_vowel_dict[s]

    if len(s) > 2 and s[:3] in tri_starts_with_vowel_dict:
        prefix = s[:3].lower()
        return tri_starts_with_vowel_dict[prefix]

    if len(s) > 1 and s[:2] in bi_starts_with_vowel_dict:
        prefix = s[:2].lower()
        return bi_starts_with_vowel_dict[prefix]

    # if len(s) > 0 and s[:1] in one_starts_with_vowel_dict:
    #     prefix = s[:1].lower()
    #     return one_starts_with_vowel_dict[prefix]
    # return 0


def create_key_from_chunk(chunk):
    chunk = re.sub('[\"]', '', chunk)
    chunk = ' '.join(chunk.split())

    return chunk

    # df = create_df(load_train())
    # bl=df[df[st_with_v].isnull()]
