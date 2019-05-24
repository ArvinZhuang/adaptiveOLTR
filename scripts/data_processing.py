import numpy as np

def process_data(file, feature_size):
    query_docid_get_feature = {}
    query_docid_get_index = {}
    query_get_docids = {}
    query_get_all_features = {}
    query_docid_get_rel= {}
    feature_set = []

    with open(file) as fin:
        for line in fin:
            cols = line.strip().split()
            query = cols[1].split(':')[1]
            docid = cols[feature_size + 4]
            relevence = int(cols[0])
            if relevence > 0:
                relevence = 1
            feature = []

            for i in range(2, 2 + feature_size):
                feature.append(float(cols[i].split(':')[1]))

            feature_set.append(feature)

            if query in query_get_docids.keys():
                query_docid_get_feature[query][docid] = feature
                query_get_docids[query].append(docid)
                #query_docid_get_index[query][docid] = len(query_get_docids[query]) - 1
                query_get_all_features[query].append(feature)
                query_docid_get_rel[query][docid] = relevence
            else:
                query_docid_get_feature[query] = {docid: feature}
                query_get_docids[query] = [docid]
                #query_docid_get_index[query] = {docid: 0}
                query_get_all_features[query]= [feature]
                query_docid_get_rel[query] = {docid: relevence}
    return query_docid_get_feature,  query_get_docids, query_get_all_features, query_docid_get_rel, feature_set


def get_query_pos_docids(file, feature_size):
    query_pos_docids = {}
    with open(file) as fin:
        for line in fin:
            cols = line.split()
            rank = float(cols[0])
            query = cols[1].split(':')[1]
            docid = cols[feature_size + 4]

            if rank > 0.0:
                if query in query_pos_docids:
                    query_pos_docids[query].append(docid)
                else:
                    query_pos_docids[query] = [docid]


    return query_pos_docids
