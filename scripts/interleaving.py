import numpy as np

def k_greedy_interleaving(list1, list2, k):
    result_list = []
    for r in range(0, 10):
        if np.random.rand() >= k:
            currentList = list1
        else:
            currentList = list2

        result_set = set(result_list)
        for i in range(0,len(currentList)):
            if currentList[i] not in result_set:
                result_list.append(currentList[i])
                break
    return result_list

def count_based_interleaving(list1, list2, rank, LSHash, new_weights):
    result_list = []

    key = LSHash._hash(LSHash.uniform_planes[0], new_weights)
    explore_rate = LSHash.hash_tables[0].get_val(key)
    LSHash.hash_tables[0].add_time(LSHash._hash(LSHash.uniform_planes[0], new_weights))
    # if explore_rate!=0:
    #     k = k/(explore_rate**0.5)

    k = 0.5/np.sqrt(np.add(explore_rate, 1))

    # if k < 1:
    #     print (k)
    # if k < 0.1:
    #     k = 0.1
    # if k > 0.5:
    #     k = 0.5
    for r in range(0, rank):
        if np.random.rand() >= k:
            currentList = list1
        else:
            currentList = list2
        result_set = set(result_list)
        for i in range(0, len(currentList)):
            if currentList[i] not in result_set:
                result_list.append(currentList[i])
                break
    return result_list, k
