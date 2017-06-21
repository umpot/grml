from preprocessing import *
from validation import *

def add_corrections_cols(df):
    def get_corrections(row):
        a = row['a']
        an = row['an']
        the = row['the']

        s = [(a, 'a'), (an, 'an'), (the, 'the')]
        s.sort(key=lambda s: s[0], reverse=True)

        proposed_correction = s[0][1]
        art_val = row[article]
        if proposed_correction == inverse_art_map[art_val]:
            return None, None
        else:
            return proposed_correction, s[0][0]

    df[tmp] = df.apply(get_corrections, axis=1)
    df[correction] = df[tmp].apply(lambda s: s[0])
    df[confidence] = df[tmp].apply(lambda s: s[1])

def submit_xgb_test():
    train_arr = load_train()
    test_arr = load_test()

    TARGET = correct_article
    df, cols = to_xgb_df(train_arr)
    df, cols = to_xgb_df(test_arr)

    train_arr = train_arr[cols]
    test_arr=test_arr[cols]

    print len(train_arr), len(test_arr)
    train_target = train_arr[TARGET]
    del train_arr[TARGET]

    test_target = test_arr[TARGET]
    del test_arr[TARGET]
    print test_target.head()

    estimator = xgb.XGBClassifier(n_estimators=250,
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

    for c in classes:
        col = inverse_art_map[c]
        test_arr[col] =proba[:,classes.index(c)]
        df.loc[test_arr.index, col] = test_arr.loc[test_arr.index, col]

    add_corrections_cols(df)
    sentences = load_test_arr()
    res = df_to_submit_array(df, sentences)

    json.dump(res, open('test_submission.json', 'w+'))

    # return res


def submit_xgb_private_test():
    train_arr = load_train()
    private_test_arr = load_private_test()

    TARGET = correct_article
    df, cols = to_xgb_df(train_arr)
    df, cols = to_xgb_df(private_test_arr)

    train_arr = train_arr[cols]
    private_test_arr=private_test_arr[cols]

    print len(train_arr), len(private_test_arr)
    train_target = train_arr[TARGET]
    del train_arr[TARGET]

    private_test_target = private_test_arr[TARGET]
    del private_test_arr[TARGET]
    print private_test_target.head()

    estimator = xgb.XGBClassifier(n_estimators=180,
                                  subsample=0.8,
                                  colsample_bytree=0.8,
                                  max_depth=5,
                                  # learning_rate=learning_rate,
                                  objective='mlogloss',
                                  nthread=-1
                                  )
    print private_test_arr.columns.values

    estimator.fit(
        train_arr, train_target,
        verbose=True
    )

    proba = estimator.predict_proba(private_test_arr)

    classes = list(estimator.classes_)
    print classes

    for c in classes:
        col = inverse_art_map[c]
        private_test_arr[col] =proba[:,classes.index(c)]
        df.loc[private_test_arr.index, col] = private_test_arr.loc[private_test_arr.index, col]

    add_corrections_cols(df)
    sentences = load_private_test_arr()
    res = df_to_submit_array(df, sentences)

    json.dump(res, open('private_test_submission.json', 'w+'))

    # return res
