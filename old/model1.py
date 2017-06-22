from evaluation import *


def process_max_ngram_freq_suff_naive(df, conf_level, max_sum=2, a_an_score=5):
    for col in [correction, confidence, solution_details]:
        if col in df:
            del col

    def get_scores(row, col1,col2):
        art = row[article]
        a_freq_suff = row[col1]
        the_freq_suff = row[col2]

        return (10.0 + a_freq_suff)/(10.0 + the_freq_suff), (10.0 + the_freq_suff)/(10.0 + a_freq_suff)



    def fix(row):
        art = row[article]
        a_art = row[indef_article]
        if row[difficult]:#row[ngram_confidence_level] < conf_cuttof
            return None, None, None
        if row[a_five_freq_suff] is not None and row[a_five_freq_suff]+ row[the_five_freq_suff]>max_sum:
            a_score, the_score = get_scores(row, a_five_freq_suff, the_five_freq_suff)
            if is_definite_article(art) and a_score>=conf_level:
                return a_art, a_score*2,5
            if is_indefinite_article(art) and the_score >= conf_level:
                return 'the', the_score*2,5
            if is_indefinite_article(art) and art != a_art and a_score>=conf_level:
                return a_art, a_an_score*2,5


        if row[a_four_freq_suff] is not None and row[a_four_freq_suff]+ row[the_four_freq_suff]>max_sum:
            a_score, the_score = get_scores(row, a_four_freq_suff, the_four_freq_suff)
            if is_definite_article(art) and a_score>=conf_level:
                return a_art, a_score*1.7,4
            if is_indefinite_article(art) and the_score >= conf_level:
                return 'the', the_score*1.7,4
            if is_indefinite_article(art) and art != a_art and a_score>=conf_level:
                return a_art, a_an_score*1.7,4

        if row[a_three_freq_suff] is not None and row[a_three_freq_suff]+ row[the_three_freq_suff]>max_sum:
            a_score, the_score = get_scores(row, a_three_freq_suff, the_three_freq_suff)
            if is_definite_article(art) and a_score>=conf_level:
                return a_art, a_score*1.3,3
            if is_indefinite_article(art) and the_score >= conf_level:
                return 'the', the_score*1.3,3
            if is_indefinite_article(art) and art != a_art and a_score>=conf_level:
                return a_art, a_an_score*1.3,3

        if row[a_bi_freq_suff] is not None and row[a_bi_freq_suff]+ row[the_bi_freq_suff]>max_sum:
            a_score, the_score = get_scores(row, a_bi_freq_suff, the_bi_freq_suff)
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