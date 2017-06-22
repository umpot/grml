import pandas as pd

from submit.src.preprocessing import *


def is_indefinite_article(a):
    return a in {'a', 'an'}


def is_definite_article(a):
    return a == 'the'

def submit_train(df):
    train_arr = load_train_arr()
    corrections = load_resource(fp_corrections_train)
    submit_arr = create_submission_arr(df, train_arr)
    return evaluate(train_arr, corrections, submit_arr)

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
                tel = None if s is None else s[2]
                if s is None or s[0] == w:
                    s = ['', float('-inf')]
                # -score, ok-prediction, is_realy_error
                data.append((-s[1], s[0] == c, c is not None, tel))

                # -1, -0.8, -0.2, ... inf, inf
    print 0.02 * len(data)

    data.sort(key=lambda x: x[0])
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


def create_submission_arr(df, sents):
    arr = [[None] * (len(x)) for x in sents]

    cols = df.columns

    def do_work(row):
        if row[correction] is not None:
            corr = row[correction]
            conf = row[confidence]
            sent_index = row[sentence_index]
            pos = row[position]
            arr[sent_index][pos] = (corr, conf, OrderedDict([(c,row[c]) for c in cols]))

    df.apply(do_work, axis=1)

    return arr

def arr_to_df(arr):
    cols = arr[0].keys()
    m = OrderedDict((c, [None if x is None else x[c] for x in arr]) for c in cols)
    df= pd.DataFrame(m)

    df['trans'] = df[article]+'->'+df[correction].apply(str)
    add_short_freq_cols(df)
    cols = ['trans', confidence, solution_details, sentence_index,suffix_ngrams, bi, three, four, five]

    return df[cols]


def process_max_ngram_freq_naive(df, conf_level, max_sum=2, a_an_score=5):
    for col in [correction, confidence, solution_details]:
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
            return None, None, None
        if row[a_five_freq] is not None and row[a_five_freq]+ row[the_five_freq]>max_sum:
            a_score, the_score = get_scores(row, a_five_freq, the_five_freq)
            if is_definite_article(art) and a_score>=conf_level:
                return a_art, a_score*2,5
            if is_indefinite_article(art) and the_score >= conf_level:
                return 'the', the_score*2,5
            if is_indefinite_article(art) and art != a_art and a_score>=conf_level:
                return a_art, a_an_score*2,5


        if row[a_four_freq] is not None and row[a_four_freq]+ row[the_four_freq]>max_sum:
            a_score, the_score = get_scores(row, a_four_freq, the_four_freq)
            if is_definite_article(art) and a_score>=conf_level:
                return a_art, a_score*1.7,4
            if is_indefinite_article(art) and the_score >= conf_level:
                return 'the', the_score*1.7,4
            if is_indefinite_article(art) and art != a_art and a_score>=conf_level:
                return a_art, a_an_score*1.7,4

        if row[a_three_freq] is not None and row[a_three_freq]+ row[the_three_freq]>max_sum:
            a_score, the_score = get_scores(row, a_three_freq, the_three_freq)
            if is_definite_article(art) and a_score>=conf_level:
                return a_art, a_score*1.3,3
            if is_indefinite_article(art) and the_score >= conf_level:
                return 'the', the_score*1.3,3
            if is_indefinite_article(art) and art != a_art and a_score>=conf_level:
                return a_art, a_an_score*1.3,3

        if row[a_bi_freq] is not None and row[a_bi_freq]+ row[the_bi_freq]>max_sum:
            a_score, the_score = get_scores(row, a_bi_freq, the_bi_freq)
            if is_definite_article(art) and a_score>=conf_level:
                return a_art, a_score,2
            if is_indefinite_article(art) and the_score >= conf_level:
                return 'the', the_score,2
            if is_indefinite_article(art) and art != a_art and a_score>=conf_level:
                return a_art, a_an_score,2


        return None, None,  None

    df[tmp] = df.apply(fix, axis=1)
    df[correction] = df[tmp].apply(lambda s: s[0])
    df[confidence] = df[tmp].apply(lambda s: s[1])
    df[solution_details] = df[tmp].apply(lambda s: s[2])

    del df[tmp]

    stats = submit_train(df)

    return arr_to_df(stats['w']), arr_to_df(stats['r'])


def process_max_ngram_freq_naive1(df, conf_level,conf_neg_level, max_sum=2, a_an_score=5):
    for col in [correction, confidence, solution_details]:
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
            return None, None, None
        if row[a_five_freq] is not None and row[a_five_freq]+ row[the_five_freq]>max_sum:
            a_score, the_score = get_scores(row, a_five_freq, the_five_freq)
            if a_score<conf_neg_level and the_score<conf_neg_level:
                return None, None, None
            if is_definite_article(art) and a_score>=conf_level:
                return a_art, a_score*2,5
            if is_indefinite_article(art) and the_score >= conf_level:
                return 'the', the_score*2,5
            if is_indefinite_article(art) and art != a_art and a_score>=conf_level:
                return a_art, a_an_score*2,5


        if row[a_four_freq] is not None and row[a_four_freq]+ row[the_four_freq]>max_sum:
            a_score, the_score = get_scores(row, a_four_freq, the_four_freq)
            if a_score<conf_neg_level and the_score<conf_neg_level:
                return None, None, None
            if is_definite_article(art) and a_score>=conf_level:
                return a_art, a_score*1.7,4
            if is_indefinite_article(art) and the_score >= conf_level:
                return 'the', the_score*1.7,4
            if is_indefinite_article(art) and art != a_art and a_score>=conf_level:
                return a_art, a_an_score*1.7,4

        if row[a_three_freq] is not None and row[a_three_freq]+ row[the_three_freq]>max_sum:
            a_score, the_score = get_scores(row, a_three_freq, the_three_freq)
            if a_score<conf_neg_level and the_score<conf_neg_level:
                return None, None, None
            if is_definite_article(art) and a_score>=conf_level:
                return a_art, a_score*1.3,3
            if is_indefinite_article(art) and the_score >= conf_level:
                return 'the', the_score*1.3,3
            if is_indefinite_article(art) and art != a_art and a_score>=conf_level:
                return a_art, a_an_score*1.3,3

        if row[a_bi_freq] is not None and row[a_bi_freq]+ row[the_bi_freq]>max_sum:
            a_score, the_score = get_scores(row, a_bi_freq, the_bi_freq)
            if a_score<conf_neg_level and the_score<conf_neg_level:
                return None, None, None
            if is_definite_article(art) and a_score>=conf_level:
                return a_art, a_score,2
            if is_indefinite_article(art) and the_score >= conf_level:
                return 'the', the_score,2
            if is_indefinite_article(art) and art != a_art and a_score>=conf_level:
                return a_art, a_an_score,2


        return None, None,  None

    df[tmp] = df.apply(fix, axis=1)
    df[correction] = df[tmp].apply(lambda s: s[0])
    df[confidence] = df[tmp].apply(lambda s: s[1])
    df[solution_details] = df[tmp].apply(lambda s: s[2])

    del df[tmp]

    stats = submit_train(df)

    return arr_to_df(stats['w']), arr_to_df(stats['r'])


# train_df = exploring_df(load_train())
# train_df = load_train()
#bad, good = process_max_ngram_freq_naive(train_df, 8, 5, 10)   63%
bad, good = process_max_ngram_freq_naive1(train_df, 10, 5, 10)