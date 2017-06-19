from sklearn.metrics import log_loss

from preprocessing import *
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt



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


def to_xgb_df(df):
    art_map = {'a':0, 'an':1, 'the':2}
    df[article]=df[article].apply(lambda s: art_map[s])
    df[correct_article]=df[correct_article].apply(lambda s: art_map[s])

    def normalized_ratio(row, col1, col2, N):
        a = row[col1]
        b = row[col2]
        if a is None:
            return None
        return (float(N)+a)/(float(N)+b)

    df['a_bi_ratio_1']=df.apply(
        lambda row: normalized_ratio(row, a_bi_freq, the_bi_freq, 1),
        axis=1
    )
    df['the_bi_ratio_1']=df.apply(
        lambda row: normalized_ratio(row, the_bi_freq, the_bi_freq, 1),
        axis=1
    )

    df['a_bi_ratio_10']=df.apply(
        lambda row: normalized_ratio(row, a_bi_freq, the_bi_freq, 10),
        axis=1
    )
    df['the_bi_ratio_10']=df.apply(
        lambda row: normalized_ratio(row, the_bi_freq, the_bi_freq, 10),
        axis=1
    )

    df['a_bi_ratio_100']=df.apply(
        lambda row: normalized_ratio(row, a_bi_freq, the_bi_freq, 100),
        axis=1
    )
    df['the_bi_ratio_100']=df.apply(
        lambda row: normalized_ratio(row, the_bi_freq, the_bi_freq, 100),
        axis=1
    )

    df['a_three_ratio_1']=df.apply(
        lambda row: normalized_ratio(row, a_three_freq, the_three_freq, 1),
        axis=1
    )
    df['the_three_ratio_1']=df.apply(
        lambda row: normalized_ratio(row, the_three_freq, the_three_freq, 1),
        axis=1
    )

    df['a_three_ratio_10']=df.apply(
        lambda row: normalized_ratio(row, a_three_freq, the_three_freq, 10),
        axis=1
    )
    df['the_three_ratio_10']=df.apply(
        lambda row: normalized_ratio(row, the_three_freq, the_three_freq, 10),
        axis=1
    )

    df['a_three_ratio_100']=df.apply(
        lambda row: normalized_ratio(row, a_three_freq, the_three_freq, 100),
        axis=1
    )
    df['the_three_ratio_100']=df.apply(
        lambda row: normalized_ratio(row, the_three_freq, the_three_freq, 100),
        axis=1
    )

    df['a_four_ratio_1']=df.apply(
        lambda row: normalized_ratio(row, a_four_freq, the_four_freq, 1),
        axis=1
    )
    df['the_four_ratio_1']=df.apply(
        lambda row: normalized_ratio(row, the_four_freq, the_four_freq, 1),
        axis=1
    )

    df['a_four_ratio_10']=df.apply(
        lambda row: normalized_ratio(row, a_four_freq, the_four_freq, 10),
        axis=1
    )
    df['the_four_ratio_10']=df.apply(
        lambda row: normalized_ratio(row, the_four_freq, the_four_freq, 10),
        axis=1
    )

    df['a_four_ratio_100']=df.apply(
        lambda row: normalized_ratio(row, a_four_freq, the_four_freq, 100),
        axis=1
    )
    df['the_four_ratio_100']=df.apply(
        lambda row: normalized_ratio(row, the_four_freq, the_four_freq, 100),
        axis=1
    )

    df['a_five_ratio_1']=df.apply(
        lambda row: normalized_ratio(row, a_five_freq, the_five_freq, 1),
        axis=1
    )
    df['the_five_ratio_1']=df.apply(
        lambda row: normalized_ratio(row, the_five_freq, the_five_freq, 1),
        axis=1
    )

    df['a_five_ratio_10']=df.apply(
        lambda row: normalized_ratio(row, a_five_freq, the_five_freq, 10),
        axis=1
    )
    df['the_five_ratio_10']=df.apply(
        lambda row: normalized_ratio(row, the_five_freq, the_five_freq, 10),
        axis=1
    )

    df['a_five_ratio_100']=df.apply(
        lambda row: normalized_ratio(row, a_five_freq, the_five_freq, 100),
        axis=1
    )
    df['the_five_ratio_100']=df.apply(
        lambda row: normalized_ratio(row, the_five_freq, the_five_freq, 100),
        axis=1
    )

    cols = [
        a_bi_freq, the_bi_freq,
        a_three_freq, the_three_freq,
        a_four_freq, the_four_freq,
        a_five_freq, the_five_freq,
        'the_bi_ratio_100',
        'a_bi_ratio_100',
        'the_bi_ratio_10',
        'a_bi_ratio_10',
        'the_bi_ratio_1',
        'a_bi_ratio_1',

        'the_three_ratio_100',
        'a_three_ratio_100',
        'the_three_ratio_10',
        'a_three_ratio_10',
        'the_three_ratio_1',
        'a_three_ratio_1',

        'the_four_ratio_100',
        'a_four_ratio_100',
        'the_four_ratio_10',
        'a_four_ratio_10',
        'the_four_ratio_1',
        'a_four_ratio_1',

        'the_five_ratio_100',
        'a_five_ratio_100',
        'the_five_ratio_10',
        'a_five_ratio_10',
        'the_five_ratio_1',
        'a_five_ratio_1',

        st_with_v,

    ]

    return cols



def perform_xgboost():
    TARGET = correct_article
    train_arr, test_arr = load_train_cv()
    cols = to_xgb_df(train_arr)
    cols = to_xgb_df(test_arr)

    train_target = train_arr[TARGET]
    del train_arr[TARGET]

    test_target = test_arr[TARGET]
    del test_arr[TARGET]

    train_arr, test_arr = train_arr[cols], test_arr[cols]

    estimator = xgb.XGBClassifier(n_estimators=10000,
                                  subsample=0.8,
                                  colsample_bytree=0.8,
                                  max_depth=5,
                                  # learning_rate=learning_rate,
                                  objective='mlogloss',
                                  nthread=-1
                                  )
    print test_arr.columns.values

    eval_set = [(train_arr, train_target), (test_arr, test_target)]
    estimator.fit(
        train_arr, train_target,
        eval_set=eval_set,
        eval_metric='mlogloss',
        verbose=True,
        early_stopping_rounds=50
    )

    xgb.plot_importance(estimator)
    plt.show()


    # plot_importance(estimator)
    # pyplot.show()

    proba = estimator.predict_proba(test_arr)
    loss = log_loss(test_target, proba)
    print loss




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

# train_df = exploring_df(load_train())
# train_df = load_train()
#bad, good = process_max_ngram_freq_naive(train_df, 8, 5, 10)   63%
# bad, good = process_max_ngram_freq_naive(train_df, 8, 5, 10)