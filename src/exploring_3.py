from preprocessing import *

bi='bi'
three = 'three'
four = 'four'
five = 'five'
N = 'N'
NNs = 'NNs'

def exploring_df(df):
    columns = [
        article,
        correct_article,
        correct,
        def_correct,

        suffix_ngrams,
        suffix_noun,
        suffix_nounS,
        the_end
    ]

    add_short_freq_cols(df)

    columns=[
        suffix_ngrams,
        suffix_noun,
        suffix_nounS,
        bi,
        three,
        four,
        five,
        N,
        NNs
    ]

    return df[columns]


def add_short_freq_cols(df, col_pairs, cols):
    for i in range(len(cols)):
        new_col = cols[i]
        col1, col2 = col_pairs[i]
        df[new_col] = df[col1].apply(str) + '/' + df[col2].apply(str)