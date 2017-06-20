# coding=utf-8
from itertools import izip_longest

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

#########################################################
#Loading data...
#########################################################
data_folder = '../../data/grml/grammarly_research_assignment/data'

fp_sentence_train = 'sentence_train'
fp_sentence_test = 'sentence_test'
fp_sentence_private_test = 'sentence_private_test'

fp_corrections_train = 'corrections_train'
fp_corrections_test = 'corrections_test'

fp_parse_train = 'parse_train'
fp_parse_test = 'parse_test'
fp_parse_private_test = 'parse_private_test'

fp_pos_tags_train, fp_pos_tags_test, fp_pos_tags_private_test = \
    'pos_tags_train', 'pos_tags_test', 'pos_tags_private_test'

fp_ngrams = 'ngrams'
fp_syntactic_ngrams = 'syntactic_ngrams'
fp_idioms = 'idioms'
fp_transcriptions = 'transcriptions'

fp_singular_nouns = 'singular_nouns'
fp_plural_nouns = 'plural_nouns'
fp_uncountable_nouns = 'uncountable_nouns'

fp_dependencies_train='dependencies_train'
fp_dependencies_test='dependencies_test'
fp_dependencies_private_test='dependencies_private_test'

def get(name):
    return os.path.join(data_folder, name)


def load_resource(name):
    with open(get(name + '.txt')) as f:
        return json.load(f, encoding='utf-8')


def load_transcription():
    return load_resource(fp_transcriptions)

def load_train_arr():
    return load_resource(fp_sentence_train)


def load_test_arr():
    return load_resource(fp_sentence_test)

def load_ngrams():
    return load_resource(fp_ngrams)


def load_syntactic_ngrams():
    return load_resource(fp_syntactic_ngrams)


#########################################################
#
#########################################################
#########################################################
#Constants
#########################################################
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
    u'\xe6',    # 'æ',
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

prev_token_tags = [u'IN',
             u'VB',
             u'VBZ',
             u'VBD',
             u'VBG',
             u'CC',
             u'RB',
             u',',
             u'VBP',
             u'VBN',
             u'PDT',
             u'NN',
             u'WRB',
             u'RP',
             u'PRP',
             u':',
             u'WP',
             u'NNS',
             u'NNP',
             u'WDT',
             u'JJ',
             u'-LRB-',
             u'``',
             u'DT']

next_token_tags=[u'NN',
                 u'JJ',
                 u'NNP',
                 u'NNS',
                 u'RB',
                 u'``',
                 u'NNPS',
                 u'JJS',
                 u'CD',
                 u'DT',
                 u'VBN',
                 u'JJR',
                 u'RBS',
                 u'RBR']




symbols_to_skip = '[\'\(\-\{\@]'  # {"'", '(', '-', '{'}
#########################################################
#
#########################################################
#########################################################
# Vowel/Consonant preprocessing...
#########################################################
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

def starts_with_vowel(s):
    if s is None:
        print 'None'
        return 0, True
    s = s.lower()
    if s in starts_with_vowel_dict:
        return starts_with_vowel_dict[s], False

    if len(s) > 2 and s[:3] in tri_starts_with_vowel_dict:
        prefix = s[:3].lower()
        return tri_starts_with_vowel_dict[prefix], False

    if len(s) > 1 and s[:2] in bi_starts_with_vowel_dict:
        prefix = s[:2].lower()
        return bi_starts_with_vowel_dict[prefix], False

    if len(s) > 0 and s[:1] in one_starts_with_vowel_dict:
        prefix = s[:1].lower()
        return one_starts_with_vowel_dict[prefix], False

    return 0, True



#########################################################
#
#########################################################
#########################################################
#Columns
#########################################################
tmp = 'tmp'

# original_chunk = 'original_chunk'
# correct_chunk = 'correct_chunk'
# definite_chunk = 'definite_chunk'
# indefinite_chunk = 'indefinite_chunk'

correct = 'correct'
def_correct = 'def_correct'
sentence = 'sentence'
full_sentence = 'full_sentence'
next_token = 'next_token'
next_token_POS = 'next_token_POS'
first_noun = 'first_noun'
raw_next_token = 'raw_next_token'
article = 'article'
correct_article = 'correct_article'
raw_suffix = 'raw_suffix'
sentence_index = 'sentence_index'

st_with_v = 'st_with_v'


difficult_vowel_detection = 'difficult_vowel_detection'
difficult_noun_detection = 'difficult_noun_detection'
difficult_suffix = 'difficult_suffix'
suffix = 'suffix'
difficult = 'difficult'
raw_next_token_POS = 'raw_next_token_POS'
raw_prev_token_POS = 'raw_prev_token_POS'
position = 'position'
correction = 'correction'
confidence = 'confidence'
solution_details = 'solution_details'

the_end = 'the_end'
suffix_ngrams = 'suffix_ngrams'
prefix_ngrams = 'prefix_ngrams'
indef_article = 'indef_article'

a_bi_chunk_suff = 'a_bi_chunk_suff'
the_bi_chunk_suff = 'the_bi_chunk_suff'
a_bi_freq_suff = 'a_bi_freq_suff'
the_bi_freq_suff = 'the_bi_freq_suff'

a_three_chunk_suff = 'a_three_chunk_suff'
the_three_chunk_suff = 'the_three_chunk_suff'
a_three_freq_suff = 'a_three_freq_suff'
the_three_freq_suff = 'the_three_freq_suff'

a_four_chunk_suff = 'a_four_chunk_suff'
the_four_chunk_suff = 'the_four_chunk_suff'
a_four_freq_suff = 'a_four_freq_suff'
the_four_freq_suff = 'the_four_freq_suff'

a_five_chunk_suff = 'a_five_chunk_suff'
the_five_chunk_suff = 'the_five_chunk_suff'
a_five_freq_suff = 'a_five_freq_suff'
the_five_freq_suff = 'the_five_freq_suff'

the_bi_freq_pref='the_bi_freq_pref'
a_bi_freq_pref='a_bi_freq_pref'
the_bi_chunk_pref='the_bi_chunk_pref'
a_bi_chunk_pref='a_bi_chunk_pref'

the_three_freq_pref= 'the_three_freq_pref'
a_three_freq_pref= 'a_three_freq_pref'
the_three_chunk_pref= 'the_three_chunk_pref'
a_three_chunk_pref= 'a_three_chunk_pref'

the_four_freq_pref='the_four_freq_pref'
a_four_freq_pref='a_four_freq_pref'
the_four_chunk_pref='the_four_chunk_pref'
a_four_chunk_pref='a_four_chunk_pref'

the_five_freq_pref='the_five_freq_pref'
a_five_freq_pref='a_five_freq_pref'
the_five_chunk_pref='the_five_chunk_pref'
a_five_chunk_pref='a_five_chunk_pref'



a_noun_chunk_freq = 'a_noun_chunk_freq'
the_noun_chunk_freq = 'the_noun_chunk_freq'

a_nounS_chunk_freq = 'a_nounS_chunk_freq'
the_nounS_chunk_freq = 'the_nounS_chunk_freq'

suffix_noun='suffix_noun'
suffix_nounS='suffix_nounS'
#########################################################
#
#########################################################
#########################################################
#Preprocessing...
#########################################################
def preprocessing_step1_general(x, sentence_index_val):
    for i, y in enumerate(x):
        article_val = y[0]
        correct_article_val = y[1]
        pos = y[2]
        if article_val not in ARTICLES:
            continue

        if correct_article_val is None:
            correct_article_val = article_val

        if i==0:
            raw_prev_token_POS_val=None
        else:
            raw_prev_token_POS_val = x[i - 1][2]

        attachment = {}
        attachment[raw_next_token] = x[i + 1][0]
        attachment[raw_next_token_POS] = x[i + 1][2]
        attachment[raw_prev_token_POS] = raw_prev_token_POS_val
        attachment[position] = i
        attachment[sentence] = x
        attachment[sentence_index] = sentence_index_val
        attachment[the_end] = ' '.join(x[z][0] for z in range(i, len(x)))
        attachment[suffix_ngrams] = [x[z][0] for z in range(i + 1, len(x))][:4]
        attachment[prefix_ngrams] = [x[z][0] for z in range(max(0, i-4), i)]
        attachment[article] = article_val
        attachment[correct_article] = correct_article_val

        y.append(attachment)

def preprocessing_step1_noun_chunks(x):
    for i, y in enumerate(x):
        article_val = y[0]
        if article_val not in ARTICLES:
            continue

        attachment = y[3]


        suffix_noun_val=[]
        suffix_nounS_val = []
        next_token_val = None
        first_noun_val = None

        found_noun = False
        found_next_token = False

        j = i
        while j < len(x):
            j += 1
            if j == len(x):
                break

            next_y = x[j]
            current_token = next_y[0]
            next_POS = next_y[2]

            if not found_next_token and current_token not in {'"', "'", ',', '*'}:
                next_token_val = current_token
                found_next_token = True

            if next_POS in NOUNS:
                if not found_noun:
                    suffix_noun_val.append(current_token)
                    found_noun=True

                suffix_nounS_val.append(current_token)
            else:
                if found_noun:
                    break
                else:
                    suffix_noun_val.append(current_token)
                    suffix_nounS_val.append(current_token)

        suffix_noun_val = None if len(suffix_noun_val)==0 else ' '.join(suffix_noun_val)
        suffix_nounS_val = None if len(suffix_nounS_val)==0 else ' '.join(suffix_nounS_val)

        attachment[suffix_noun]=suffix_noun_val
        attachment[suffix_nounS] = suffix_nounS_val
        attachment[next_token]=next_token_val


def preprocessing_step1(x, sentence_index):
    preprocessing_step1_general(x, sentence_index)
    preprocessing_step1_noun_chunks(x)


def get_sentence_plain(x):
    return ' '.join([y[0] for y in x])


def preprocessin_step2(arr):
    l = []
    ngrams = load_ngrams()

    for x in arr:
        for y in x:
            if len(y) > 3:
                l.append((y, x))

    m = {
        article: [x[0][3][article] for x in l],
        correct_article: [x[0][3][correct_article] for x in l],
        next_token: [x[0][3][next_token] for x in l],
        raw_next_token: [x[0][3][raw_next_token] for x in l],
        position: [x[0][3][position] for x in l],
        full_sentence: [x[1] for x in l],
        sentence: [get_sentence_plain(x[1]) for x in l],
        sentence_index: [x[0][3][sentence_index] for x in l],
        the_end: [x[0][3][the_end] for x in l],
        suffix_ngrams: [x[0][3][suffix_ngrams] for x in l],
        prefix_ngrams: [x[0][3][prefix_ngrams] for x in l],
        suffix_noun:[x[0][3][suffix_noun] for x in l],
        suffix_nounS:[x[0][3][suffix_nounS] for x in l],
        raw_next_token_POS:[x[0][3][raw_next_token_POS] for x in l],
        raw_prev_token_POS:[x[0][3][raw_prev_token_POS] for x in l]
    }

    df = pd.DataFrame(m)

    df[st_with_v] = df[next_token].apply(starts_with_vowel)
    df[difficult_vowel_detection] = df[st_with_v].apply(lambda s: s[1])
    df[st_with_v] = df[st_with_v].apply(lambda s: s[0])
    df[indef_article] = df[st_with_v].apply(lambda s: 'an' if s == 1 else 'a')

    def get_n_gram_suff_freq(row, n):
        ss = row[suffix_ngrams]
        if len(ss) + 1 < n:
            return None, None
        indef_art = row[indef_article]
        suf = ' '.join(ss[:n - 1])

        return ngrams.get(indef_art + ' ' + suf, 0), ngrams.get('the ' + suf, 0)

    def get_n_suff_freq(row, col):
        indef_art = row[indef_article]
        suf = row[col]
        if suf is None:
            return None, None
        return ngrams.get(indef_art + ' ' + suf, 0), ngrams.get('the ' + suf, 0)

    def get_n_gram_chunks(row, n):
        ss = row[suffix_ngrams]
        if len(ss) + 1 < n:
            return None, None
        indef_art = row[indef_article]
        suf = ' '.join(ss[:n - 1])

        return indef_art + ' ' + suf, 'the ' + suf

    def get_n_gram_prefix_chunks(row, n):
        ss = row[prefix_ngrams]
        if len(ss) + 1 < n:
            return None, None
        indef_art = row[indef_article]
        pref = ' '.join(ss[-(n-1):])

        return pref + ' ' + indef_art, pref + ' the'


    def get_n_pref_freq(row, n):
        ss = row[prefix_ngrams]
        if len(ss) + 1 < n:
            return None, None
        indef_art = row[indef_article]
        pref = ' '.join(ss[-(n-1):])

        return ngrams.get(pref + ' ' + indef_art, 0), ngrams.get(pref + ' the', 0)


    df[tmp] = df.apply(lambda row: get_n_gram_chunks(row, 2), axis=1)
    df[a_bi_chunk_suff] = df[tmp].apply(lambda s: s[0])
    df[the_bi_chunk_suff] = df[tmp].apply(lambda s: s[1])
    df[tmp] = df.apply(lambda row: get_n_gram_suff_freq(row, 2), axis=1)
    df[a_bi_freq_suff] = df[tmp].apply(lambda s: s[0])
    df[the_bi_freq_suff] = df[tmp].apply(lambda s: s[1])

    df[tmp] = df.apply(lambda row: get_n_gram_chunks(row, 3), axis=1)
    df[a_three_chunk_suff] = df[tmp].apply(lambda s: s[0])
    df[the_three_chunk_suff] = df[tmp].apply(lambda s: s[1])
    df[tmp] = df.apply(lambda row: get_n_gram_suff_freq(row, 3), axis=1)
    df[a_three_freq_suff] = df[tmp].apply(lambda s: s[0])
    df[the_three_freq_suff] = df[tmp].apply(lambda s: s[1])

    df[tmp] = df.apply(lambda row: get_n_gram_chunks(row, 4), axis=1)
    df[a_four_chunk_suff] = df[tmp].apply(lambda s: s[0])
    df[the_four_chunk_suff] = df[tmp].apply(lambda s: s[1])
    df[tmp] = df.apply(lambda row: get_n_gram_suff_freq(row, 4), axis=1)
    df[a_four_freq_suff] = df[tmp].apply(lambda s: s[0])
    df[the_four_freq_suff] = df[tmp].apply(lambda s: s[1])

    df[tmp] = df.apply(lambda row: get_n_gram_chunks(row, 5), axis=1)
    df[a_five_chunk_suff] = df[tmp].apply(lambda s: s[0])
    df[the_five_chunk_suff] = df[tmp].apply(lambda s: s[1])
    df[tmp] = df.apply(lambda row: get_n_gram_suff_freq(row, 5), axis=1)
    df[a_five_freq_suff] = df[tmp].apply(lambda s: s[0])
    df[the_five_freq_suff] = df[tmp].apply(lambda s: s[1])


    df[tmp] = df.apply(lambda row: get_n_gram_prefix_chunks(row, 2), axis=1)
    df[a_bi_chunk_pref] = df[tmp].apply(lambda s: s[0])
    df[the_bi_chunk_pref] = df[tmp].apply(lambda s: s[1])
    df[tmp] = df.apply(lambda row: get_n_pref_freq(row, 2), axis=1)
    df[a_bi_freq_pref] = df[tmp].apply(lambda s: s[0])
    df[the_bi_freq_pref] = df[tmp].apply(lambda s: s[1])

    df[tmp] = df.apply(lambda row: get_n_gram_prefix_chunks(row, 2), axis=1)
    df[a_three_chunk_pref] = df[tmp].apply(lambda s: s[0])
    df[the_three_chunk_pref] = df[tmp].apply(lambda s: s[1])
    df[tmp] = df.apply(lambda row: get_n_pref_freq(row, 2), axis=1)
    df[a_three_freq_pref] = df[tmp].apply(lambda s: s[0])
    df[the_three_freq_pref] = df[tmp].apply(lambda s: s[1])

    df[tmp] = df.apply(lambda row: get_n_gram_prefix_chunks(row, 2), axis=1)
    df[a_four_chunk_pref] = df[tmp].apply(lambda s: s[0])
    df[the_four_chunk_pref] = df[tmp].apply(lambda s: s[1])
    df[tmp] = df.apply(lambda row: get_n_pref_freq(row, 2), axis=1)
    df[a_four_freq_pref] = df[tmp].apply(lambda s: s[0])
    df[the_four_freq_pref] = df[tmp].apply(lambda s: s[1])

    df[tmp] = df.apply(lambda row: get_n_gram_prefix_chunks(row, 2), axis=1)
    df[a_five_chunk_pref] = df[tmp].apply(lambda s: s[0])
    df[the_five_chunk_pref] = df[tmp].apply(lambda s: s[1])
    df[tmp] = df.apply(lambda row: get_n_pref_freq(row, 2), axis=1)
    df[a_five_freq_pref] = df[tmp].apply(lambda s: s[0])
    df[the_five_freq_pref] = df[tmp].apply(lambda s: s[1])



    df[tmp] = df.apply(lambda row: get_n_suff_freq(row, suffix_noun), axis=1)
    df[a_noun_chunk_freq] = df[tmp].apply(lambda s: s[0])
    df[the_noun_chunk_freq] = df[tmp].apply(lambda s: s[1])

    df[tmp] = df.apply(lambda row: get_n_suff_freq(row, suffix_nounS), axis=1)
    df[a_nounS_chunk_freq] = df[tmp].apply(lambda s: s[0])
    df[the_nounS_chunk_freq] = df[tmp].apply(lambda s: s[1])
    # df[difficult_noun_detection] = df[noun].isnull()
    # df[difficult_suffix] = (df[raw_suffix] != df[suffix])
    # df[difficult] = \
    #     df[difficult_vowel_detection] | \
    #     df[difficult_noun_detection] | \
    #     df[difficult_suffix]
    df[difficult] = False

    df[def_correct] = df[correct_article] == 'the'
    df[correct] = df[article] == df[correct_article]

    return df


def add_dummy_cols_df(df, col, vals):
    new_cols = []
    df[col] = df[col].apply(lambda s: s if s in vals else None)
    new_cols+=['{}_{}'.format(col, v) for v in vals]

    return pd.get_dummies(df, columns=[col]), new_cols




def load_train():
    sent_train = load_resource(fp_sentence_train)
    corr_train = load_resource(fp_corrections_train)
    pos_train = load_resource(fp_pos_tags_train)
    print len(sent_train) == len(corr_train)
    res = [zip(sent_train[i], corr_train[i], pos_train[i]) for i in range(len(sent_train))]
    res = [[list(y) for y in x] for x in res]

    for i, x in enumerate(res):
        preprocessing_step1(x, i)

    return preprocessin_step2(res)

def split_arr(df, c):
    msk = np.random.rand(len(df)) < c
    a=[]
    b=[]
    for m, x in zip(msk, df):
        if m:
            a.append(x)
        else:
            b.append(x)
    return a,b


def load_train_cv_old(seed=42, c=0.66):
    sent_train = load_resource(fp_sentence_train)
    corr_train = load_resource(fp_corrections_train)
    pos_train = load_resource(fp_pos_tags_train)
    print len(sent_train) == len(corr_train)
    res = [zip(sent_train[i], corr_train[i], pos_train[i]) for i in range(len(sent_train))]
    res = [[list(y) for y in x] for x in res]

    for i, x in enumerate(res):
        preprocessing_step1(x, i)

    np.random.seed(seed)

    a,b = split_arr(res, c)
    a=preprocessin_step2(a)
    b=preprocessin_step2(b)

    return a,b

def load_test():
    sent_test = load_resource(fp_sentence_test)
    corr_test = load_resource(fp_corrections_test)
    pos_test = load_resource(fp_pos_tags_test)
    print len(sent_test) == len(corr_test)
    res = [zip(sent_test[i], corr_test[i], pos_test[i]) for i in range(len(sent_test))]
    res = [[list(y) for y in x] for x in res]

    for i, x in enumerate(res):
        preprocessing_step1(x, i)

    return preprocessin_step2(res)
#########################################################
#
#########################################################

# train_df = exploring_df(load_train())