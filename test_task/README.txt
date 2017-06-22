1)Model performance:
    - Target score of 3-fold-cv on train ≈ 74%
    - Target score on test ≈ 75%

2)Overview of the model(short):
    - Xgboost, approximately 60 features(mostly based on n-grams/syntactic-n-grams frequencies)
    - I used only three files: n-grams, syntactic-n-grams, transcription

3)Running code:
    You should change DATA_FOLDER in preprocessing.py to path to 'grammarly_research_assignment/data'

4)Overview of the model:
    - Obtained possible indefinite article for each case (indef_candidate)
    - Calculated frequencies of 'definite ngram' and 'indefinite ngram':
            - Frequencies of 'the next_token_1 ... next_token_k' and
                'indef_candidate next_token_1 ... next_token_k' for
                k=1,2,3,4
            - Frequencies of 'previous_token_1 ...previous_token_k the' and
                'previous_token_1 ...previous_token_k indef_candidate'
                k=1,2,3,4
    - Calculated frequencies for syntactic-n-grams:
        parent_token/parentPOS:type:the:DET
        parent_token/parentPOS:type:indef_candidate:DET
        Where parent_token/parentPOS it's a value/POS-tag of parent of article in dependency tree
    - Calculated frequencies ratios : (frequency_of_definite_ngram +prior)/(frequency_of_indefinite_ngram +prior)
        for prior = 1,10,100 (Some sort of smoothing)
    - Added some basic features: 'article' (identity of the article: 'a'=0, 'an'=1, 'the'=2),
        if_next_token_starts_with_vowel(0/1)
    - Trained Xgboost classifier with target = correct article (multi-class classification with objective=logloss)
    - Selected article with the highest probability

5)Code overview:
    - There are 5 files:
        - preprocessing.py code for preprocessing(loading data, creating features, transforming to DataFrame etc)
        - syntactic_ngrams.py code for manipulation with synctactic-n-grams
        - validation.py code for performance evaluation(CV, parameter tuning, exploring model's errors, evaluation metric etc)
        - submitting.py code for creation submission file
        - private_test_submission.json file with corrections for private test
