from sklearn.metrics import log_loss

from preprocessing import *
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt


art_map = {'a':0, 'an':1, 'the':2}
inverse_art_map={0:'a', 1:'an', 2:'the'}


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
    # add_short_freq_cols(df)
    cols = ['trans', confidence, solution_details, sentence_index,suffix_ngrams]

    return df[cols]


def to_xgb_df(df):

    df[article]=df[article].apply(lambda s: art_map[s])
    df[correct_article]=df[correct_article].apply(lambda s: art_map[s])

    def normalized_ratio(row, col1, col2, prior):
        a = row[col1]
        b = row[col2]
        if a is None:
            return None
        return (float(prior) + a) / (float(prior) + b)

    def create_normalized_ratio_col(d, col1, col2, new_col, prior):
        d[new_col] = d.apply(lambda row: normalized_ratio(row, col1, col2, prior), axis=1)


    col_pairs=[
        (a_bi_freq_suff, the_bi_freq_suff, 'bi_freq_suff'),
        (a_three_freq_suff, the_three_freq_suff, 'three_freq_suff'),
        (a_four_freq_suff, the_four_freq_suff, 'four_freq_suff'),
        (a_five_freq_suff, the_five_freq_suff, 'five_freq_suff'),

        (a_bi_freq_pref, the_bi_freq_pref, 'bi_freq_pref'),
        (a_three_freq_pref, the_three_freq_pref, 'three_freq_pref'),
        (a_four_freq_pref, the_four_freq_pref, 'four_freq_pref'),
        (a_five_freq_pref, the_five_freq_pref, 'five_freq_pref')
    ]

    frequencies_cols = [x[0] for x in col_pairs]+[x[1] for x in col_pairs]

    for prior in [1,10,100]:
        for a_col, the_col, name in col_pairs:
            new_col = 'a_the_{}_p{}'.format(name, prior)
            create_normalized_ratio_col(df, a_col, the_col, new_col, prior)
            frequencies_cols.append(new_col)

            new_col = 'the_a_{}_p{}'.format(name, prior)
            create_normalized_ratio_col(df, the_col, a_col, new_col, prior)
            frequencies_cols.append(new_col)

    cols_to_exclude = [
        'the_bi_freq_pref', 'the_three_freq_pref',
        'the_four_freq_pref' ,'the_five_freq_pref',
        'a_bi_freq_pref', 'a_three_freq_pref',
        'a_four_freq_pref', 'a_five_freq_pref'
    ]

    frequencies_cols = list(set(frequencies_cols).difference(set(cols_to_exclude)))

    cols = frequencies_cols + [st_with_v,article, correct_article]

    return cols


def create_splits(df, cv, seed=42):
    s_indexes = set(df[sentence_index])
    np.random.seed(seed)
    m = {j: np.random.randint(0, cv) for j in s_indexes}
    df['fold'] = df[sentence_index].apply(lambda s: m[s])
    res = []
    for f in range(cv):
        train = df[df['fold']!=f]
        test = df[df['fold']==f]
        res.append((train, test))

    return res



def add_corrections_cols(df):
    def get_corrections(row):
        a = row['a']
        an = row['an']
        the = row['the']

        s = [(a, 'a'), (an, 'an'), (the, 'the')]
        s.sort(key=lambda s: s[0], reverse=True)

        proposed_correction = s[0][1]
        art_val = row[article]
        if proposed_correction == art_val:
            return None, None
        else:
            return proposed_correction, s[0][0]

    df[tmp] = df.apply(get_corrections, axis=1)
    df[correction] = df[tmp].apply(lambda s: s[0])
    df[confidence] = df[tmp].apply(lambda s: s[1])


def df_to_submit_array(df, sentences):
    res = [[None]*len(x) for x in sentences]

    def collect_corrections_info(row):
        correction_val = row[correction]
        confidence_val = row[confidence]
        sentence_index_val = row[sentence_index]
        position_val  = row[position]

        if correction_val is None:
            return

        res[sentence_index_val][position_val] = (correction_val, confidence_val)

    df.apply(collect_corrections_info, axis=1)

    # res = [(k,v) for k,v in res.iteritems()]
    # res.sort(key=lambda s: s[0])
    # res = [x[1] for x in res]

    return res


def submit_xgb_test():
    train_arr = load_train()
    test_arr = load_test()
    df = test_arr.copy()

    TARGET = correct_article
    cols = to_xgb_df(train_arr)
    cols = to_xgb_df(test_arr)

    train_arr = train_arr[cols]
    test_arr=test_arr[cols]

    print len(train_arr), len(test_arr)
    train_target = train_arr[TARGET]
    del train_arr[TARGET]

    test_target = test_arr[TARGET]
    del test_arr[TARGET]
    print test_target.head()

    estimator = xgb.XGBClassifier(n_estimators=100,
                                  subsample=0.8,
                                  colsample_bytree=0.8,
                                  max_depth=5,
                                  # learning_rate=learning_rate,
                                  objective='mlogloss',
                                  nthread=-1
                                  )
    print test_arr.columns.values

    estimator.fit(
        train_arr, train_target,
        verbose=True
    )

    proba = estimator.predict_proba(test_arr)

    classes = list(estimator.classes_)
    print classes

    for c in  classes:
        col = inverse_art_map[c]
        test_arr[col] =proba[:,classes.index(c)]
        df.loc[test_arr.index, col] = test_arr.loc[test_arr.index, col]

    add_corrections_cols(df)
    sentences = load_test_arr()
    res = df_to_submit_array(df, sentences)

    json.dump(res, open('test_submition.json', 'w+'))

    return res


def submit_xgb_out_of_fold_pred(df):
    # df = load_train()
    create_out_of_fold_xgb_predictions(df)
    add_corrections_cols(df)

    stats = submit_train(df)

    return arr_to_df(stats['w']), arr_to_df(stats['r'])



def create_out_of_fold_xgb_predictions(df):
    # df = load_train()
    TARGET = correct_article
    cols = to_xgb_df(df)
    losses = []
    for train_arr, test_arr in create_splits(df, 3):
        train_arr = train_arr[cols]
        test_arr=test_arr[cols]

        print len(train_arr), len(test_arr)
        train_target = train_arr[TARGET]
        del train_arr[TARGET]

        test_target = test_arr[TARGET]
        del test_arr[TARGET]

        # train_arr, test_arr = train_arr[cols], test_arr[cols]

        estimator = xgb.XGBClassifier(n_estimators=100,
                                      subsample=0.8,
                                      colsample_bytree=0.8,
                                      max_depth=5,
                                      # learning_rate=learning_rate,
                                      objective='mlogloss',
                                      nthread=-1
                                      )
        print test_arr.columns.values

        estimator.fit(
            train_arr, train_target,
            verbose=True
        )


        proba = estimator.predict_proba(test_arr)

        classes = list(estimator.classes_)
        print classes

        for c in  classes:
            col = inverse_art_map[c]
            test_arr[col] =proba[:,classes.index(c)]
            df.loc[test_arr.index, col] = test_arr.loc[test_arr.index, col]

        loss = log_loss(test_target, proba)
        losses.append(loss)
        print loss



def perform_xgboost_cv(df):
    # df = load_train()
    TARGET = correct_article
    cols = to_xgb_df(df)
    losses = []
    for train_arr, test_arr in create_splits(df, 3):
        train_arr = train_arr[cols]
        test_arr=test_arr[cols]

        print len(train_arr), len(test_arr)
        train_target = train_arr[TARGET]
        del train_arr[TARGET]

        test_target = test_arr[TARGET]
        del test_arr[TARGET]

        # train_arr, test_arr = train_arr[cols], test_arr[cols]

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
        classes = list(estimator.classes_)
        print classes


        xgb.plot_importance(estimator)
        plt.show()

        proba = estimator.predict_proba(test_arr)
        loss = log_loss(test_target, proba)
        losses.append(loss)
        print loss


cols=['a_bi_freq_suff' 'a_three_freq_suff' 'a_four_freq_suff' 'a_five_freq_suff'
      'a_bi_freq_pref' 'a_three_freq_pref' 'a_four_freq_pref' 'a_five_freq_pref'
      'the_bi_freq_suff' 'the_three_freq_suff' 'the_four_freq_suff'
      'the_five_freq_suff' 'the_bi_freq_pref' 'the_three_freq_pref'
      'the_four_freq_pref' 'the_five_freq_pref' 'a_the_bi_freq_suff_p1'
      'the_a_bi_freq_suff_p1' 'a_the_three_freq_suff_p1'
      'the_a_three_freq_suff_p1' 'a_the_four_freq_suff_p1'
      'the_a_four_freq_suff_p1' 'a_the_five_freq_suff_p1'
      'the_a_five_freq_suff_p1' 'a_the_bi_freq_pref_p1' 'the_a_bi_freq_pref_p1'
      'a_the_three_freq_pref_p1' 'the_a_three_freq_pref_p1'
      'a_the_four_freq_pref_p1' 'the_a_four_freq_pref_p1'
      'a_the_five_freq_pref_p1' 'the_a_five_freq_pref_p1'
      'a_the_bi_freq_suff_p10' 'the_a_bi_freq_suff_p10'
      'a_the_three_freq_suff_p10' 'the_a_three_freq_suff_p10'
      'a_the_four_freq_suff_p10' 'the_a_four_freq_suff_p10'
      'a_the_five_freq_suff_p10' 'the_a_five_freq_suff_p10'
      'a_the_bi_freq_pref_p10' 'the_a_bi_freq_pref_p10'
      'a_the_three_freq_pref_p10' 'the_a_three_freq_pref_p10'
      'a_the_four_freq_pref_p10' 'the_a_four_freq_pref_p10'
      'a_the_five_freq_pref_p10' 'the_a_five_freq_pref_p10'
      'a_the_bi_freq_suff_p100' 'the_a_bi_freq_suff_p100'
      'a_the_three_freq_suff_p100' 'the_a_three_freq_suff_p100'
      'a_the_four_freq_suff_p100' 'the_a_four_freq_suff_p100'
      'a_the_five_freq_suff_p100' 'the_a_five_freq_suff_p100'
      'a_the_bi_freq_pref_p100' 'the_a_bi_freq_pref_p100'
      'a_the_three_freq_pref_p100' 'the_a_three_freq_pref_p100'
      'a_the_four_freq_pref_p100' 'the_a_four_freq_pref_p100'
      'a_the_five_freq_pref_p100' 'the_a_five_freq_pref_p100' 'st_with_v'
      'article']




# train_df = exploring_df(load_train())
# train_df = load_train()
#bad, good = process_max_ngram_freq_naive(train_df, 8, 5, 10)   63%
# bad, good = process_max_ngram_freq_naive(train_df, 8, 5, 10)