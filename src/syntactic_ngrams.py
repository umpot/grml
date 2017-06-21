from collections import OrderedDict, defaultdict, Counter

child_index = 'childIndex'
child_token = 'child_token'
parent_token = 'parent_token'
child_pos = 'child_POS'
parent_pos = 'parent_POS'

def preprocess_dependency_tree(dep_trees, sentences, postag_sentences):
    for j, dep in enumerate(dep_trees):
        y = defaultdict(list)
        for x in dep:
            childIndex = x[child_index]
            y[childIndex].append(x)
        dep_trees[j] = y


    for dep, toks, pos in zip(dep_trees, sentences, postag_sentences):
        for j in range(len(toks)):
            # t = tok
            # p=pos[j]
            ll = dep[j]
            if len(ll)==0:
                continue

            for x in ll:
                x[child_token] = toks[j]
                x[child_pos] = pos[j]

                headIndex = x['headIndex']
                if headIndex==-1:
                    x[parent_token] = None
                    x[parent_pos]=None
                else:
                    x[parent_token] = toks[headIndex]
                    x[parent_pos]=pos[headIndex]


def create_sn_gram_for_article(items, art_val):
    if len(items)==0:
        return None
    item = items[0]

    c_tok = item[child_token]
    c_POS = item[child_pos]
    tp=item['type']
    p_tok = item[parent_token]
    p_POS = item[parent_pos]
    p_tok = '' if p_tok is None else p_tok
    p_POS = '' if p_POS is None else p_POS
    return '{}/{}:{}:{}/{}'.format(p_tok, p_POS, tp, art_val, c_POS)
