import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from scipy import stats

def percentilesigma(y):
    return np.percentile(y,q=68.3)

predictions, y  = np.genfromtxt("../good_data/encaps_pred_data.txt",unpack=True)

min_energy = 0.003
max_energy = 340
bin_edge = np.logspace(np.log10(min_energy),np.log10(max_energy),20)
rel_error = np.abs((y-predictions)/y)
perc_sigma, bins_p, binnumber_p = stats.binned_statistic(y,rel_error,statistic=percentilesigma,bins=bin_edge)
mean, bins_m, binnumber_m = stats.binned_statistic(y,rel_error,statistic='mean',bins=bin_edge)

bin_m = (bins_m[:-1]+bins_m[1:])/2
bin_p = (bins_p[:-1]+bins_p[1:])/2
plt.plot(bin_p,perc_sigma,'rx',label='68.3 percentile')
plt.plot(bin_m,mean,'b.',label='mean')
plt.legend()
plt.xscale('log')
plt.xlabel("Energy / TeV")
plt.show()
