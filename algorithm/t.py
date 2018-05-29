import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import functions as func
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error



predictions, y  = np.genfromtxt("good_data/encaps_pred_data.txt",unpack=True)

print(r2_score(predictions,y))

bin_edges = np.logspace(np.log10(0.003),np.log10(340),4)
func.plot_R2_per_bin(predictions,y,bin_edges)
plt.show()
plt.close()
