import numpy as np
import os

def ndcg_at_k(query_pos_docids, query_result_list, query_docid_get_rel, k):
    ndcg = 0.0
    num_query = 0

    for query in query_docid_get_rel.keys():
        try:
            pos_docid_set = set(query_pos_docids[query])
        except:
            num_query += 1
            continue
        dcg = 0.0
        for i in range(0,k):
            if i < len(query_result_list[query]):
                docid = query_result_list[query][i]
                relevence = query_docid_get_rel[query][docid]
                if i == 0 :
                    dcg += (2 ** relevence - 1)
                else:
                    dcg += ((2**relevence-1)/ np.log2(i+1))
            else:
                dcg = 0.0
                
        rel_set = []
        for docid in pos_docid_set:
            rel_set.append(query_docid_get_rel[query][docid])
        rel_set = sorted(rel_set,reverse=True)
        n = len(pos_docid_set) if len(pos_docid_set)<k else k
        
        idcg = 0
        for i in range(n):
            if i == 0:
                idcg += (2**rel_set[i]-1)
            else:
                idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 1))
       
        ndcg += (dcg/idcg)
        num_query += 1
    return ndcg/float(num_query)


def ndcg(query_pos_docids, query_result_list, query_docid_get_rel):
    ndcg = 0.0
    num_query = 0

    for query in query_docid_get_rel.keys():
        try:
            pos_docid_set = set(query_pos_docids[query])
        except:
            num_query += 1
            continue
        dcg = 0.0
        for i in range(0, len(query_result_list[query])):
            docid = query_result_list[query][i]
            relevence = query_docid_get_rel[query][docid]
            if i == 0:
                dcg += (2 ** relevence - 1)
            else:
                dcg += ((2 ** relevence - 1) / np.log2(i + 1))

        rel_set = []
        for docid in pos_docid_set:
            rel_set.append(query_docid_get_rel[query][docid])
        rel_set = sorted(rel_set, reverse=True)
        n = len(pos_docid_set)

        idcg = 0
        for i in range(n):
            if i == 0:
                idcg += (2 ** rel_set[i] - 1)
            else:
                idcg += ((2 ** rel_set[i] - 1) / np.log2(i + 1))

        ndcg += (dcg / idcg)
        num_query += 1
    return ndcg / float(num_query)


def evl_at_k(weights, feature_set, workdir):
    feature_set = np.array(feature_set)
    weights = np.array(weights)
    score = np.dot(feature_set, weights)

    with open('score_set.txt', 'w') as f:
        for item in score:
            f.write("%s\n" % item)

    os.system('perl Eval-Score-3.0.pl ' + workdir + ' score_set.txt evl_result.txt 0')


    with open('evl_result.txt') as fin:
        for line in fin:
            cols = line.strip().split()
            if len(cols) != 0 and cols[0] == 'NDCG:':
                print(cols)
