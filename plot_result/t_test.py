import pickle
import numpy as np
from scipy.stats import ttest_ind



def get_data(dataset, click_model, rate,fold):
    with open("{}/fold{}/{}_{}_fold{}_run1_ndcg.txt".format(dataset,fold,click_model,rate,fold), "rb") as fp:
        data = pickle.load(fp)
    data = np.array(data)
    for r in range(24):
        with open("{}/fold{}/{}_{}_fold{}_run{}_ndcg.txt".format(dataset,fold,click_model,rate,fold,r + 2), "rb") as fp:
            l = pickle.load(fp)
            data = np.vstack((data, l))

    data = data.T
    return data[-1]


data_adp= get_data('np2003_results','per','adprate', 1)
for f in range(2,6):
    data_temp = get_data('np2003_results','per','adprate', f)
    data_adp = np.append(data_adp,data_temp)



data = get_data('np2003_results','per','rate05', 1)
for f in range(2,6):
    data_temp = get_data('np2003_results','per','rate05', f)

    data = np.append(data,data_temp)

stat, p = ttest_ind(data, data_adp)
if p < 0.01:
    print('**')
elif p <0.05:
    print('*')
else:
    print(p)
