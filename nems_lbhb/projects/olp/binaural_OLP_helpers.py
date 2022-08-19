def label_pair_types(eps, parm):
    twostims = eps[eps['name'].str.count('-0-1') == 2].copy()
    pairs = [bb.replace(' ', '')+ff.replace(' ', '') for bb, ff in parm['pairs']]
    names = [ep.split('_')[1].split('-')[0]+ep.split('_')[2].split('-')[0]
             for ep in twostims.name]
    pair_labels = [str(pairs.index(nm)) for nm in names]

    type_labels = [ep.split('_')[1].split('-')[3]+ep.split('_')[2].split('-')[3]
                    for ep in twostims.name]
    labels = [pl+tl for pl, tl in zip(pair_labels, type_labels)]
    twostims['type'] = labels
    return twostims