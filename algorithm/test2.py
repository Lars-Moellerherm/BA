import itertools
import functions as func
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py as h5
import scipy as sc
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier


prediction, truth = np.genfromtxt("good_data/encaps_pred_data.txt", unpack=True)

min_energy = 0.0001
max_energy = 1
bin_edges = np.logspace(np.log10(min_energy),np.log10(max_energy),50)

func.plot_hist2d(prediction, truth, min_energy, max_energy, bin_edges)
plt.savefig("plots/hist2d_tiefe_E.jpg")
plt.close()

min_energy = 100
max_energy = max(truth)+50
bin_edges = np.logspace(np.log10(min_energy),np.log10(max_energy),50)

func.plot_hist2d(prediction, truth, min_energy, max_energy, bin_edges)
plt.xlim(100,max_energy)
plt.ylim(100,max_energy)
plt.savefig("plots/hist2d_hohe_E.jpg")
plt.close()

gammas = h5.File("../data/3_gen/gammas.hdf5","r")

# Converting to pandas
gamma_array_df = pd.DataFrame(data=dict(gammas['array_events']))
gamma_runs_df = pd.DataFrame(data=dict(gammas['runs']))
gamma_telescope_df = pd.DataFrame(data=dict(gammas['telescope_events']))

gamma_array_df = gamma_array_df
gamma_runs_df = gamma_runs_df
gamma_telescope_df = gamma_telescope_df


#merging of array and telescope data and shuffle of proton and gamma
gamma_merge = pd.merge(gamma_array_df,gamma_telescope_df,on=list(["array_event_id",'run_id']))
gamma_merge = gamma_merge.set_index(['run_id','array_event_id'])
#there are some nan in width the needed to be deleted
gamma_merge = gamma_merge.dropna(axis=0)
data = gamma_merge

def poisson(k,lamb,C):
    return lamb**k/sc.misc.factorial(k)*np.exp(-lamb) + C

energy =  data['mc_energy']
N = energy.shape[0]
hist,bin = np.histogram(energy,range=(0,0.2))
params, pcov = sc.optimize.curve_fit(poisson,bin[1:],hist/N,p0=(3,5))
x = np.linspace(0,0.2,60)
plt.plot(x, poisson(x,*params),'r-')
plt.plot(bin[1:], hist/N, 'b.')
plt.show()
plt.close()


minimum= min(energy)
maximum = max(energy)
bin_edges = np.linspace(minimum,0.2)
plt.hist(energy,bins=bin_edges)
plt.title("Energyspektrum mit Min: %.5f und Max: %.5f" % (minimum,maximum))
plt.xlabel("Energy in TeV")
plt.savefig("plots/energiespektrum.jpg")
plt.close()
