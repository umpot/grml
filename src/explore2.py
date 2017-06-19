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
    return load_resource(fp_transcriptions)


def evaluate(text, correct, submission):
    print len(text), len(correct), len(submission)
    # with open(text_file) as f:
    #     text = json.load(f)
    # with open(correct_file) as f:
    #     correct = json.load(f)
    # with open(submission_file) as f:
    #     submission = json.load(f)
    data = []
    for sent, cor, sub in izip_longest(text, correct, submission):
        for w, c, s in izip_longest(sent, cor, sub):
            if w in ['a', 'an', 'the']:
                tel = {'w': w, 'c': c, 's': s}
                if s is None or s[0] == w:
                    s = ['', float('-inf')]
                # -score, ok-prediction, is_realy_error
                data.append((-s[1], s[0] == c, c is not None, tel))

                # -1, -0.8, -0.2, ... inf, inf
    print 0.02 * len(data)

    data.sort()
    fp2 = 0
    fp = 0
    tp = 0
    all_mistakes = sum(x[2] for x in data)  # num of ALL incorrect
    print 'all_mistakes {}'.format(all_mistakes)
    score = 0
    acc = 0
    counter = 0
    stats = {'w': [], 'r': []}
    stop = False
    for _, c, r, tel in data:
        fp2 += not c  # wrong correction
        fp += not r  # realy errors count
        tp += c  # right correction

        if not stop:
            if not c:
                stats['w'].append(tel)
            if c:
                stats['r'].append(tel)

        acc = max(acc, 1 - (0. + fp + all_mistakes - tp) / len(data))
        if fp2 * 1. / len(data) <= 0.02:
            score = tp * 1. / all_mistakes
        else:
            stop = True
    print 'target score = %.2f %%' % (score * 100)
    print 'accuracy (just for info) = %.2f %%' % (acc * 100)

    return stats


def submit_test(df):
    test = load_resource(fp_sentence_test)
    corrections = load_resource(fp_corrections_test)


def submit_train(df):
    train_arr = load_train_arr()
    corrections = load_resource(fp_corrections_train)
    submit_arr = create_submission_arr(df, train_arr)
    return evaluate(train_arr, corrections, submit_arr)


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


def load_train_arr():
    return load_resource(fp_sentence_train)


def load_train_raw():
    sent_train = load_resource(fp_sentence_train)
    corr_train = load_resource(fp_corrections_train)
    pos_train = load_resource(fp_pos_tags_train)
    print len(sent_train) == len(corr_train)
    res = [zip(sent_train[i], corr_train[i], pos_train[i]) for i in range(len(sent_train))]
    res = [[list(y) for y in x] for x in res]

    for i, x in enumerate(res):
        process_sentence_array(x, i)

    return process_array(res)


def load_train():
    return filter_difficult(load_train_raw())


def load_test():
    return filter_difficult(load_test_raw())


def load_test_raw():
    sent_test = load_resource(fp_sentence_test)
    corr_test = load_resource(fp_corrections_test)
    pos_test = load_resource(fp_pos_tags_test)
    print len(sent_test) == len(corr_test)
    res = [zip(sent_test[i], corr_test[i], pos_test[i]) for i in range(len(sent_test))]
    res = [[list(y) for y in x] for x in res]

    for i, x in enumerate(res):
        process_sentence_array(x, i)

    return process_array(res)


def load_ngrams():
    return load_resource(fp_ngrams)


def load_syntactic_ngrams():
    return load_resource(fp_syntactic_ngrams)


def to_line(l):
    return ' '.join(l)


tmp = 'tmp'

original_chunk = 'original_chunk'
correct_chunk = 'correct_chunk'
definite_chunk = 'definite_chunk'
indefinite_chunk = 'indefinite_chunk'

noun = 'noun'
correct = 'correct'
def_correct = 'def_correct'
sentence = 'sentence'
next_token = 'next_token'
next_token_POS = 'next_token_POS'

st_with_v = 'st_with_v'

def_ngram_count = 'def_ngram_count'
indef_ngram_count = 'indef_ngram_count'
def_ngram_count_ratio = 'def_ngram_count_ratio'
indef_ngram_count_ratio = 'in_def_ngram_count_ratio'
ngram_confidence_level = 'ngram_confidence_level'

raw_next_token = 'raw_next_token'
article = 'article'
correct_article = 'correct_article'
raw_suffix = 'raw_suffix'
correct_article = correct_article
sentence_index = 'sentence_index'
difficult_vowel_detection = 'difficult_vowel_detection'
difficult_noun_detection = 'difficult_noun_detection'
difficult_suffix = 'difficult_suffix'
suffix = 'suffix'
difficult = 'difficult'
raw_next_token_POS = 'raw_next_token_POS'
position = 'position'
correction = 'correction'
confidence = 'confidence'
the_end = 'the_end'
suffix_ngrams = 'suffix_ngrams'
indef_article = 'indef_article'

a_bi_chunk = 'a_bi_chunk'
the_bi_chunk = 'the_bi_chunk'
a_bi_freq = 'a_bi_freq'
the_bi_freq = 'the_bi_freq'

a_three_chunk = 'a_three_chunk'
the_three_chunk = 'the_three_chunk'
a_three_freq = 'a_three_freq'
the_three_freq = 'the_three_freq'

a_four_chunk = 'a_four_chunk'
the_four_chunk = 'the_four_chunk'
a_four_freq = 'a_four_freq'
the_four_freq = 'the_four_freq'

a_five_chunk = 'a_five_chunk'
the_five_chunk = 'the_five_chunk'
a_five_freq = 'a_five_freq'
the_five_freq = 'the_five_freq'


def process_sentence_array(x, s_index):
    for i, y in enumerate(x):
        art = y[0]
        corr_article = y[1]
        pos = y[2]
        if art not in ARTICLES:
            continue

        if corr_article is None:
            corr_article = art

        attachment = {}
        attachment[raw_next_token] = x[i + 1][0]
        attachment[raw_next_token_POS] = x[i + 1][2]
        attachment[position] = i
        attachment[sentence] = x
        attachment[sentence_index] = s_index
        attachment[the_end] = ' '.join(x[z][0] for z in range(i, len(x)))
        attachment[suffix_ngrams] = [x[z][0] for z in range(i + 1, len(x))][:4]

        noun_val = None
        nxt_token = None
        nxt_token_POS = None
        found_next_token = False
        r_suffix = []

        j = i
        while j < len(x):
            j += 1
            if j == len(x):
                break

            next_y = x[j]
            t = next_y[0]
            next_pos = next_y[2]

            r_suffix.append(t)

            if not found_next_token and t not in {'"', "'", ',', '*'}:
                nxt_token = t
                nxt_token_POS = next_pos
                found_next_token = True

            if next_pos in NOUNS:
                noun_val = t
                break

        attachment[noun] = noun_val
        attachment[article] = art
        attachment[correct_article] = corr_article
        attachment[raw_suffix] = ' '.join(r_suffix)
        attachment[next_token] = nxt_token
        attachment[next_token_POS] = nxt_token_POS

        y.append(attachment)


def process_array(arr):
    l = []
    ngrams = load_ngrams()

    def get_suffix(ss):
        ss = ss.split()
        ss = filter(lambda s: s not in {'"', '(', ')', ',', '&', '*'}, ss)
        return ' '.join(ss)

    def get_sentence_plain(x):
        return ' '.join([y[0] for y in x])

    def create_indef_chunk(s, starts_with_vowel):
        return 'an ' + s if starts_with_vowel == 1 else 'a ' + s

    for x in arr:
        for y in x:
            if len(y) > 3:
                l.append((y, x))

    m = {
        article: [x[0][3][article] for x in l],
        correct_article: [x[0][3][correct_article] for x in l],
        noun: [x[0][3][noun] for x in l],
        raw_suffix: [x[0][3][raw_suffix] for x in l],
        'next_token': [x[0][3]['next_token'] for x in l],
        'next_token_POS': [x[0][3]['next_token_POS'] for x in l],
        raw_next_token: [x[0][3][raw_next_token] for x in l],
        'raw_next_token_POS': [x[0][3]['raw_next_token_POS'] for x in l],
        'position': [x[0][3]['position'] for x in l],
        'full_sentence': [x[1] for x in l],
        'sentence': [get_sentence_plain(x[1]) for x in l],
        'sentence_index': [x[0][3]['sentence_index'] for x in l],
        suffix: [get_suffix(x[0][3][raw_suffix]) for x in l],
        the_end: [x[0][3][the_end] for x in l],
        suffix_ngrams: [x[0][3][suffix_ngrams] for x in l],

    }

    df = pd.DataFrame(m)
    columns = [
        article,
        correct_article,
        correct,
        def_correct,
        def_ngram_count_ratio,
        indef_ngram_count_ratio,
        definite_chunk,
        def_ngram_count,
        indefinite_chunk,
        indef_ngram_count,
        sentence,
        difficult,
        ngram_confidence_level,
        sentence_index,
        position,
        st_with_v,
        suffix,
        the_end,
        indef_article,

        a_bi_chunk,
        the_bi_chunk,
        a_bi_freq,
        the_bi_freq,

        a_three_chunk,
        the_three_chunk,
        a_three_freq,
        the_three_freq,

        a_four_chunk,
        the_four_chunk,
        a_four_freq,
        the_four_freq,

        a_five_chunk,
        the_five_chunk,
        a_five_freq,
        the_five_freq
        # 'article',
        # correct_article,
        # 'full_sentence',
        # 'next_token',
        # 'next_token_POS',
        # 'noun',
        # 'position',
        # raw_next_token,
        # 'raw_next_token_POS',
        # raw_suffix,
        # 'st_with_v',
        # sentence_index,
        # sentence,
        # difficult_vowel_detection,
        # difficult_suffix,
        # difficult_noun_detection,
        # difficult
    ]

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

    def get_n_gram_chunks(row, n):
        ss = row[suffix_ngrams]
        if len(ss) + 1 < n:
            return None, None
        indef_art = row[indef_article]
        suf = ' '.join(ss[:n - 1])

        return indef_art + ' ' + suf, 'the ' + suf

    df[tmp] = df.apply(lambda row: get_n_gram_chunks(row, 2), axis=1)
    df[a_bi_chunk] = df[tmp].apply(lambda s: s[0])
    df[the_bi_chunk] = df[tmp].apply(lambda s: s[1])
    df[tmp] = df.apply(lambda row: get_n_gram_suff_freq(row, 2), axis=1)
    df[a_bi_freq] = df[tmp].apply(lambda s: s[0])
    df[the_bi_freq] = df[tmp].apply(lambda s: s[1])

    df[tmp] = df.apply(lambda row: get_n_gram_chunks(row, 3), axis=1)
    df[a_three_chunk] = df[tmp].apply(lambda s: s[0])
    df[the_three_chunk] = df[tmp].apply(lambda s: s[1])
    df[tmp] = df.apply(lambda row: get_n_gram_suff_freq(row, 3), axis=1)
    df[a_three_freq] = df[tmp].apply(lambda s: s[0])
    df[the_three_freq] = df[tmp].apply(lambda s: s[1])

    df[tmp] = df.apply(lambda row: get_n_gram_chunks(row, 4), axis=1)
    df[a_four_chunk] = df[tmp].apply(lambda s: s[0])
    df[the_four_chunk] = df[tmp].apply(lambda s: s[1])
    df[tmp] = df.apply(lambda row: get_n_gram_suff_freq(row, 4), axis=1)
    df[a_four_freq] = df[tmp].apply(lambda s: s[0])
    df[the_four_freq] = df[tmp].apply(lambda s: s[1])

    df[tmp] = df.apply(lambda row: get_n_gram_chunks(row, 5), axis=1)
    df[a_five_chunk] = df[tmp].apply(lambda s: s[0])
    df[the_five_chunk] = df[tmp].apply(lambda s: s[1])
    df[tmp] = df.apply(lambda row: get_n_gram_suff_freq(row, 5), axis=1)
    df[a_five_freq] = df[tmp].apply(lambda s: s[0])
    df[the_five_freq] = df[tmp].apply(lambda s: s[1])

    df[difficult_noun_detection] = df[noun].isnull()
    df[difficult_suffix] = (df[raw_suffix] != df[suffix])
    df[difficult] = \
        df[difficult_vowel_detection] | \
        df[difficult_noun_detection] | \
        df[difficult_suffix]

    df[original_chunk] = df[article] + ' ' + df[suffix]
    df[correct_chunk] = df[correct_article] + ' ' + df[suffix]
    df[definite_chunk] = 'the ' + df[suffix]
    df[indefinite_chunk] = \
        df.apply(lambda s: create_indef_chunk(s[suffix], s[st_with_v]), axis=1)

    df[def_ngram_count] = df[definite_chunk].apply(lambda s: ngrams.get(s, 0))
    df[indef_ngram_count] = df[indefinite_chunk].apply(lambda s: ngrams.get(s, 0))
    df[def_ngram_count_ratio] = df[def_ngram_count] / (1 + df[indef_ngram_count])
    df[indef_ngram_count_ratio] = df[indef_ngram_count] / (1 + df[def_ngram_count])
    df[ngram_confidence_level] = \
        df.apply(lambda s: 1 + max(s[def_ngram_count], s[indef_ngram_count]), axis=1)

    df[def_correct] = df[correct_article] == 'the'
    df[correct] = df[article] == df[correct_article]

    return df[columns]


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


def filter_difficult(df):
    return df[~df[difficult]]


def is_indefinite_article(a):
    return a in {'a', 'an'}


def is_definite_article(a):
    return a == 'the'

def process_max_ngram_freq_naive(df, conf_level, max_sum=2, a_an_score=5):
    for col in [correction, confidence]:
        if col in df:
            del col

    def get_scores(row, col1,col2):
        art = row[article]
        a_freq = row[col1]
        the_freq = row[col2]

        return (10.0 + a_freq)/(10.0 + the_freq), (10.0 + the_freq)/(10.0 + a_freq)



    def fix(row):
        art = row[article]
        a_art = row[indef_article]
        if row[difficult]:#row[ngram_confidence_level] < conf_cuttof
            return None, None
        if row[a_five_freq] is not None and row[a_five_freq]+ row[the_five_freq]>max_sum:
            a_score, the_score = get_scores(row, a_five_freq, the_five_freq)
            if is_definite_article(art) and a_score>=conf_level:
                return a_art, a_score*2
            if is_indefinite_article(art) and the_score >= conf_level:
                return 'the', the_score*2
            if is_indefinite_article(art) and art != a_art and a_score>=conf_level:
                return a_art, a_an_score*2


        if row[a_four_freq] is not None and row[a_four_freq]+ row[the_four_freq]>max_sum:
            a_score, the_score = get_scores(row, a_four_freq, the_four_freq)
            if is_definite_article(art) and a_score>=conf_level:
                return a_art, a_score*1.7
            if is_indefinite_article(art) and the_score >= conf_level:
                return 'the', the_score*1.7
            if is_indefinite_article(art) and art != a_art and a_score>=conf_level:
                return a_art, a_an_score*1.7

        if row[a_three_freq] is not None and row[a_three_freq]+ row[the_three_freq]>max_sum:
            a_score, the_score = get_scores(row, a_three_freq, the_three_freq)
            if is_definite_article(art) and a_score>=conf_level:
                return a_art, a_score*1.3
            if is_indefinite_article(art) and the_score >= conf_level:
                return 'the', the_score*1.3
            if is_indefinite_article(art) and art != a_art and a_score>=conf_level:
                return a_art, a_an_score*1.3

        if row[a_bi_freq] is not None and row[a_bi_freq]+ row[the_bi_freq]>max_sum:
            a_score, the_score = get_scores(row, a_bi_freq, the_bi_freq)
            if is_definite_article(art) and a_score>=conf_level:
                return a_art, a_score
            if is_indefinite_article(art) and the_score >= conf_level:
                return 'the', the_score
            if is_indefinite_article(art) and art != a_art and a_score>=conf_level:
                return a_art, a_an_score


        return None, None

    df[tmp] = df.apply(fix, axis=1)
    df[correction] = df[tmp].apply(lambda s: s[0])
    df[confidence] = df[tmp].apply(lambda s: s[1])

    del df[tmp]

    stats = submit_train(df)

    for n in ['r', 'w']:
        ss = stats[n]
        m = {
            'cor': [x['c'] for x in ss],
            'w': [x['w'] for x in ss],
            'cand': [x['s'][0] for x in ss],
            'the_count': [x['s'][2] for x in ss],
            'a_count': [x['s'][3] for x in ss],
            'score': [x['s'][1] for x in ss],
            'suffix': [x['s'][4] for x in ss],
            the_end: [x['s'][5] for x in ss]
        }

        f = pd.DataFrame(m)
        f['trans'] = f['w'] + ' -> ' + f['cand']
        f = f[['cor', 'trans', 'suffix', 'score', 'the_count', 'a_count', the_end]]

        stats[n] = f

    return stats['w'], stats['r']



def process_fix_def_naive(df, the_cuttof, a_cuttof, conf_cuttof, a_val=10, an_val=10, a_the_val=10):
    for col in [correction, confidence]:
        if col in df:
            del col

    def fix(row):
        if row[ngram_confidence_level] < conf_cuttof or row[difficult]:
            return None, None

        art = row[article]

        the_ratio = (20.0 + row[def_ngram_count]) / (20 + row[indef_ngram_count])
        a_ratio = (20.0 + row[indef_ngram_count]) / (20 + row[def_ngram_count])

        if is_indefinite_article(art):
            if the_ratio > the_cuttof:
                return 'the', the_ratio

            if art == 'a' and row[st_with_v] == 1 and a_ratio > a_the_val:
                return 'an', a_val

            if art == 'an' and row[st_with_v] == 0 and a_ratio > a_the_val:
                return 'a', an_val

        if is_definite_article(art) and a_ratio > a_cuttof:
            return 'an' if row[st_with_v] == 1 else 'a', a_ratio

        return None, None

    df[tmp] = df.apply(fix, axis=1)
    df[correction] = df[tmp].apply(lambda s: s[0])
    df[confidence] = df[tmp].apply(lambda s: s[1])

    del df[tmp]

    stats = submit_train(df)

    for n in ['r', 'w']:
        ss = stats[n]
        m = {
            'cor': [x['c'] for x in ss],
            'w': [x['w'] for x in ss],
            'cand': [x['s'][0] for x in ss],
            'the_count': [x['s'][2] for x in ss],
            'a_count': [x['s'][3] for x in ss],
            'score': [x['s'][1] for x in ss],
            'suffix': [x['s'][4] for x in ss],
            the_end: [x['s'][5] for x in ss]
        }

        f = pd.DataFrame(m)
        f['trans'] = f['w'] + ' -> ' + f['cand']
        f = f[['cor', 'trans', 'suffix', 'score', 'the_count', 'a_count', the_end]]

        stats[n] = f

    return stats['w'], stats['r']


def create_submission_arr(df, sents):
    arr = [[None] * (len(x)) for x in sents]

    def do_work(row):
        if row[correction] is not None:
            corr = row[correction]
            conf = row[confidence]
            def_c = row[def_ngram_count]
            infef_c = row[indef_ngram_count]
            sent_index = row[sentence_index]
            pos = row[position]
            arr[sent_index][pos] = (corr, conf, def_c, infef_c, row[suffix], row[the_end])

    df.apply(do_work, axis=1)

    return arr


# train_df = filter_difficult(load_train())
# train_df = load_train_raw()
# process_fix_def_naive(train_df)
# arr = create_submission_arr(train_df, load_train_arr())
# test_df = filter_difficult(process_array(load_test()))
# bl=df[df[st_with_v].isnull()]
# bad, good = process_fix_def_naive(train_df, 1, 10, 10)

# df = train_df[
#     [a_bi_chunk, the_bi_chunk, a_bi_freq, the_bi_freq, a_three_chunk, the_three_chunk, a_three_freq, the_three_freq,
#      a_four_chunk, the_four_chunk, a_four_freq, the_four_freq, a_five_chunk, the_five_chunk, a_five_freq,
#      the_five_freq]]
# bad, good = process_max_ngram_freq_naive(train_df, 5)