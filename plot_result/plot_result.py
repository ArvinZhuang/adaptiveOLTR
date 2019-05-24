import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from scipy import mean


def get_data(rate,fold):
    with open("OHSUMED_results/fold{}/infor_{}_fold{}_run1_ndcg.txt".format(fold,rate,fold), "rb") as fp:
        data01 = pickle.load(fp)
    data01 = np.array(data01)
    for r in range(24):
        with open("OHSUMED_results/fold{}/infor_{}_fold{}_run{}_ndcg.txt".format(fold,rate,fold,r + 2), "rb") as fp:
            l = pickle.load(fp)
            data01 = np.vstack((data01, l))

    data01 = data01.T
    data01_mean = np.mean(data01, axis=1)
    data01_std_err = sem(data01, axis=1)
    data01_h = data01_std_err * t.ppf((1 + 0.95) / 2, 25 - 1)
    data01_low = np.subtract(data01_mean, data01_h)
    data01_high = np.add(data01_mean, data01_h)
    return data01_mean, data01_low, data01_high

data01_mean, data01_low, data01_high = get_data('rate01', 1)
for f in range(2,6):
    data_mean, data_low, data_high = get_data('rate01', f)
    data01_mean += data_mean
    data01_low += data_low
    data01_high += data_high
data01_mean /= 5
data01_low /= 5
data01_high /= 5
print("rate 0.1:", data01_mean[-1])


data02_mean, data02_low, data02_high = get_data('rate02', 1)
for f in range(2,6):
    data_mean, data_low, data_high = get_data('rate02', f)
    data02_mean += data_mean
    data02_low += data_low
    data02_high += data_high
data02_mean /= 5
data02_low /= 5
data02_high /= 5
print("rate 0.2:", data02_mean[-1])

data05_mean, data05_low, data05_high = get_data('rate05', 1)
for f in range(2,6):
    data_mean, data_low, data_high = get_data('rate05', f)
    data05_mean += data_mean
    data05_low += data_low
    data05_high += data_high
data05_mean /= 5
data05_low /= 5
data05_high /= 5
print("rate 0.5:", data05_mean[-1])

#data0_mean, data0_low, data0_high = get_data('rate0', 1)
# for f in range(2,4):
#     data_mean, data_low, data_high = get_data('rate0', f)
#     data0_mean += data_mean
#     data0_low += data_low
#     data0_high += data_high
# data0_mean /= 1
# data0_low /= 1
# data0_high /= 1


data_adp_mean, data_adp_low, data_adp_high = get_data('adprate', 1)
for f in range(2,6):
    data_mean, data_low, data_high = get_data('adprate', f)
    data_adp_mean += data_mean
    data_adp_low += data_low
    data_adp_high += data_high
data_adp_mean /= 5
data_adp_low /= 5
data_adp_high /= 5
print("rate adaptive:", data_adp_mean[-1])


plt.figure(1)

plt.plot(range(len(data01_mean)), data01_mean, color = 'red', alpha = 1)
plt.fill_between(range(len(data01_mean)), data01_low, data01_high, color = 'red', alpha = 0.2)

plt.plot(range(len(data02_mean)), data02_mean, color = 'blue', alpha = 1)
plt.fill_between(range(len(data02_mean)), data02_low, data02_high, color = 'blue', alpha = 0.2)

plt.plot(range(len(data05_mean)), data05_mean, color = 'yellow', alpha = 1)
plt.fill_between(range(len(data05_mean)), data05_low, data05_high, color = 'yellow', alpha = 0.2)

plt.plot(range(len(data_adp_mean)), data_adp_mean, color = 'black', alpha = 1)
plt.fill_between(range(len(data_adp_mean)), data_adp_low, data_adp_high, color = 'black', alpha = 0.2)

# plt.plot(range(len(data_adp_mean)), data0_mean, color = 'green', alpha = 1)
# plt.fill_between(range(len(data_adp_mean)), data0_low, data0_high, color = 'green', alpha = 0.2)

plt.ylabel('NDCG')
plt.xlabel('EPOCH')
plt.legend(('k=0.1','k=0.2','k=0.5', 'adaptive k'),  loc='lower right')
plt.show()





