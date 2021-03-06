from itertools import izip_longest
import json


def evaluate(text_file, correct_file, submission_file):
    with open(text_file) as f:
        text = json.load(f)
    with open(correct_file) as f:
        correct = json.load(f)
    with open(submission_file) as f:
        submission = json.load(f)
    data = []
    for sent, cor, sub in izip_longest(text, correct, submission):
        for w, c, s in izip_longest(sent, cor, sub):
            if w in ['a', 'an', 'the']:
                if s is None or s[0] == w:
                    s = ['', float('-inf')]
                #-score, ok-prediction, is_realy_error
                data.append((-s[1], s[0] == c, c is not None))
                #-1, -0.8, -0.2, ... inf, inf
    data.sort()
    fp2 = 0
    fp = 0
    tp = 0
    all_mistakes = sum(x[2] for x in data)#num of ALL incorrect
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


# if __name__ == '__main__':
#     evaluate('data/sentence_test.txt', 'data/corrections_test.txt', 'data/submission_test.txt')

text_file='../../data/grml/grammarly_research_assignment/data/sentence_test.txt'
correct_file='../../data/grml/grammarly_research_assignment/data/corrections_test.txt'
submission_file='../../grml/src/test_submission.json'
evaluate(text_file, correct_file, submission_file)