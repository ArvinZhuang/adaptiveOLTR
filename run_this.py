from scripts import data_processing as dp, eval_tool as et, interleaving as il
from scripts.clickModel import clickModel
from scripts.DBGD import DBGD
from scripts.lshash import LSHash
import pickle


def run(training_set, test_set, FEATURE_SIZE, EPOCH, f, r, exploration, click_model, momentum=False, count_base=False):
    query_docid_get_feature, query_get_docids, query_get_all_features, query_docid_get_rel, feature_set = dp.process_data(
        training_set, FEATURE_SIZE)
    query_pos_docids = dp.get_query_pos_docids(training_set, FEATURE_SIZE)
    dbgd = DBGD(1, 0.01, FEATURE_SIZE, query_get_docids, query_docid_get_feature, query_get_all_features)
    cm = clickModel(query_pos_docids, click_model)
    lsh = LSHash(32, FEATURE_SIZE)

    test_query_docid_get_feature, test_query_get_docids, test_query_get_all_features, test_query_docid_get_rel, test_feature_set = dp.process_data(
        test_set, FEATURE_SIZE)
    test_query_pos_docids = dp.get_query_pos_docids(test_set, FEATURE_SIZE)
    test_dbgd = DBGD(1, 0.01, FEATURE_SIZE, test_query_get_docids, test_query_docid_get_feature,
                     test_query_get_all_features)

    ndcg_scores = []

    for i in range(0, EPOCH):
        for query in query_pos_docids.keys():
            # print("qid:", query)
            list1 = dbgd.get_query_result_list(query, dbgd.weights)

            unit_vector = dbgd.sample_unit_vector()
            new_weights = dbgd.sample_new_weights(unit_vector)

            list2 = dbgd.get_query_result_list(query, new_weights)

            if count_base:
                interList, k = il.count_based_interleaving(list1, list2, 10, lsh, new_weights)
            else:
                interList = il.k_greedy_interleaving(list1, list2, exploration)

            clicked_doc = cm.simulate(query, interList)

            if dbgd.compare(list1, list2, interList, clicked_doc):
                # if momentum:
                #     dbgd.update_weights_with_momentum(unit_vector,beta = 0.9)
                # else：
                dbgd.update_weights(unit_vector)

        all_result = test_dbgd.get_all_query_result_list(dbgd.weights)
        ndcg = et.ndcg_at_k(test_query_pos_docids, all_result, test_query_docid_get_rel, 10)
        ndcg_scores.append(ndcg)
        print("fold:" + str(f) + " runs:" + str(r) + " interation: " + str(i) + " NDCG: " + str(ndcg))
    final_weight = dbgd.weights

    # for test eval scores
    # et.evl_at_k(dbgd.weights, feature_set, workdir)

    return ndcg_scores, final_weight


if __name__ == "__main__":
    FEATURE_SIZE = 45
    EPOCH = 1000

    for f in range(5):
        training_set = "dataset/Fold{}/train.txt".format(f + 1)
        test_set = "dataset/Fold{}/test.txt".format(f + 1)
        for r in range(25):
            ndcg_scores, final_weight = run(training_set, test_set, FEATURE_SIZE, EPOCH, f + 1, r + 1, exploration=0.1,
                                            click_model="informational", momentum=False,
                                            count_base=False)
            with open("results/fold{}/infor_rate01_fold{}_run{}_ndcg.txt".format(f + 1, f + 1, r + 1),
                      "wb") as fp:
                pickle.dump(ndcg_scores, fp)
            with open("results/fold{}/infor_rate01_fold{}_run{}_final_weight.txt".format(f + 1, f + 1, r + 1),
                      "wb") as fp:
                pickle.dump(final_weight, fp)
