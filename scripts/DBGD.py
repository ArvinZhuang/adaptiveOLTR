import numpy as np
from scipy.linalg import norm

class DBGD:
    def __init__(self, step_size, learning_rate, num_features, query_get_docids,
                 query_docid_get_feature, query_get_all_features):
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.feature_size= num_features
        self.query_get_docids = query_get_docids
        self.query_docid_get_feature = query_docid_get_feature
        self.query_get_all_features = query_get_all_features

        #random initialize weight
        unit_vector = np.random.randn(self.feature_size)
        unit_vector /= norm(unit_vector)
        self.weights = unit_vector * 0.01

        #for momentum test
        self.vd = 0


    """
    features: list, all query-doc pair features for one query
    return: np array, the score of every docment
    """
    def get_score(self, features, weights):
        features = np.array(features)
        weights = np.array(weights)
        score = np.dot(features, weights)
        return score.tolist()

    """
    return:
        dic:
            key: query id
            value: sorted docid accroding to the score computed by the model
    """
    def get_all_query_result_list(self, weights):
        query_result_list = {}

        for query in self.query_get_docids.keys():
            #listwise ranking with linear model
            docid_list = list(self.query_docid_get_feature[query].keys())
            score_list = self.get_score(self.query_get_all_features[query], weights)

            docid_score_list = zip(docid_list, score_list)
            docid_score_list = sorted(docid_score_list, key=lambda x: x[1], reverse=True)
            (docid, socre) = docid_score_list[0]
            query_result_list[query] = [docid]
            for i in range(1, len(docid_list)):
                (docid, socre) = docid_score_list[i]
                query_result_list[query].append(docid)

        return query_result_list

    """
    input:
        weights: the model weights
        query: query id
    return:
        the sorted docid of the given query
    """
    def get_query_result_list(self, query, weights):
        #listwise ranking with linear model
        query_result_list = []
        docid_list = list(self.query_docid_get_feature[query].keys())

        score_list = self.get_score(self.query_get_all_features[query], weights)
        docid_score_list = zip(docid_list, score_list)

        docid_score_list = sorted(docid_score_list, key=lambda x: x[1], reverse=True)
        
        for i in range(0, len(docid_list)):
            (docid, socre) = docid_score_list[i]
            query_result_list.append(docid)
        return query_result_list



    def sample_unit_vector(self):
        unit_vector = np.random.randn(self.feature_size)
        vector_norms = np.sum(unit_vector ** 2) ** (1. / 2)
        unit_vector = unit_vector/vector_norms
        return unit_vector


    
    def sample_new_weights(self,unit_vector):
          new_weights = self.weights + self.step_size * unit_vector
          return new_weights


    def update_weights(self, unit_vector):
        self.weights = self.weights + self.learning_rate * unit_vector
        
    def update_weights_with_momentum(self,unit_vector,beta=0.9):
        self.vd = beta*self.vd + (1-beta) * unit_vector
        self.weights = self.weights + self.learning_rate * self.vd
        
        
        
    def compare(self, list1, list2, interList, clicked_doc):
        if len(clicked_doc) == 0:
            #print("draw")
            return False
        else:
            dmax = clicked_doc[-1]
            v = min(list1.index(dmax), list2.index(dmax))
            c1 = 0
            c2 = 0
            for i in range(len(clicked_doc)):
                if clicked_doc[i] in list1[:v+1]:
                    c1 += 1
                if clicked_doc[i] in list2[:v+1]:
                    c2 += 1
            n1 = len(set(list1[:len(interList)]) & set(interList))
            n2 = len(set(list2[:len(interList)]) & set(interList))
            if n2 == 0:
                return False
            c2 = (n1/n2) * c2 #compensate for bias
            return c1 < c2
