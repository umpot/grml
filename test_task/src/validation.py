from itertools import izip_longest

from sklearn.metrics import log_loss

from preprocessing import *
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt



def evaluate(text, correct, submission):
    """
    Evaluation function from the task
    :param text:
    :param correct:
    :param submission:
    :return:
    """
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


def submit_train(df):
    train_arr = load_train_arr()
    corrections = load_resource(fp_corrections_train)
    submit_arr = create_submission_arr(df, train_arr)
    return evaluate(train_arr, corrections, submit_arr)


def create_submission_arr(df, sents):
    """
    Creates submission
    :param df:
    :param sents:
    :return:
    """
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

def arr_to_explore_errors_df(arr):
    """
    Transorms array to DataFrame with detailed information
    :param arr:
    :return:
    """
    cols = arr[0].keys()
    print cols
    m = OrderedDict((c, [None if x is None else x[c] for x in arr]) for c in cols)
    df= pd.DataFrame(m)

    df['trans'] = df[article]+'->'+df[correction].apply(str)#.apply(lambda s: inverse_art_map[s])
    # add_short_freq_cols(df)
    cols = [article, 'trans', 'a', 'an', 'the',
            difficult_vowel_detection, indef_article, suffix_ngrams, sentence_index]

    return df



def create_splits(df, cv, seed=42):
    """
    Creates random splits of DataFrame
    :param df:
    :param cv:
    :param seed:
    :return:
    """
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
    """
    Adds columns with correction-candidate + correction-confidence
    :param df:
    :return:
    """
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
    """
    Transforms DataFrame to submission format
    :param df:
    :param sentences:
    :return:
    """
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



def explore_xgb_errors(df, n_estimators):
    """
    Performs Xgboost prediction and returns data-frames
     with detailed information about the most right\wrong predictions
    :param df:
    :param n_estimators:
    :return:
    """
    df = create_out_of_fold_xgb_predictions(df, n_estimators)
    add_corrections_cols(df)

    stats = submit_train(df)

    return arr_to_explore_errors_df(stats['w']), arr_to_explore_errors_df(stats['r'])



def create_out_of_fold_xgb_predictions(df, n_estimators):
    """
    Creates xgboost's predictions
    :param df:
    :param n_estimators:
    :return:
    """
    # df = load_train()
    TARGET = correct_article
    df_cp = df.copy()
    df, cols = to_xgb_df(df)
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

        estimator = xgb.XGBClassifier(n_estimators=n_estimators,
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
            df_cp.loc[test_arr.index, col] = test_arr.loc[test_arr.index, col]

        loss = log_loss(test_target, proba)
        losses.append(loss)
        print loss

    return df_cp

def eval_target_score(arts, probs, labels):
    """
    Evaluates target score(i.e. recall at 2% level of FP)
    :param arts:
    :param probs:
    :param labels:
    :return:
    """
    sz=len(arts)
    all_mistakes = sum(arts[j]!=labels[j] for j in range(sz))
    data=[]
    for j in range(sz):
        p = probs[j]#probs
        a=arts[j]#current article
        correct_a = labels[j]#correct article
        p = [(i, p[i]) for i in range(len(p))]
        p.sort(key=lambda s: s[1], reverse=True)
        s = p[0][0]
        conf = p[0][1]
        conf = float('-inf') if a==s else conf
        c = s ==correct_a

        data.append((-conf, c, correct_a!=a))

    data.sort()
    fp2 = 0
    fp = 0
    tp = 0
    score = 0
    acc = 0
    for _, c, r in data:
        fp2 += not c # wrong correction
        fp += not r#realy errors count
        tp += c#right correction
        acc = max(acc, 1 - (0. + fp + all_mistakes - tp) / len(data))
        if fp2 * 1. / len(data) <= 0.02:
            score = tp * 1. / all_mistakes
    print 'target score = %.2f %%' % (score * 100)
    print 'accuracy (just for info) = %.2f %%' % (acc * 100)


    return 'target_score'  , -score


def perform_xgboost_cv(df):
    """
    Performs Cross Validation with Early Stoppings based on target-score
    :param df:
    :return:
    """
    TARGET = correct_article
    df, cols = to_xgb_df(df)
    losses = []
    for train_arr, test_arr in create_splits(df, 3):
        train_arr = train_arr[cols]
        test_arr=test_arr[cols]

        def get_articles(labels):
            if len(labels)==len(train_arr):
                return list(train_arr[article])
            elif len(labels) == len(test_arr):
                return list(test_arr[article])
            raise

        def target_score_objective(preds, dtrain):
            labels = dtrain.get_label()
            arts = get_articles(labels)

            return eval_target_score(arts, preds, labels)

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
            eval_metric=target_score_objective,#'mlogloss'
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

