import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bias_resolution import plot_bias_resolution as br_plot
import functions as func
import h5py as h5


predictions, truth = np.genfromtxt("good_data/encaps_pred_data.txt", unpack=True)
prediction_mean, truth_mean = np.genfromtxt("good_data/encaps_pred_mean_data.txt", unpack=True)

df = pd.DataFrame({'prediction':predictions, 'truth':truth})
df2 = pd.DataFrame({'prediction':prediction_mean, 'truth':truth_mean})
min_energy = 0.003
max_energy = 340
bin_edge = np.logspace(np.log10(min_energy),np.log10(max_energy),20)

ax_bias, ax_res =br_plot(df=df,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',mark='o')
br_plot(df=df2,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',mark='s')
ax_bias.set_ylabel("Bias",color="C0")
ax_res.set_ylabel('Resolution',color="C1")
ax_bias.set_xlabel(r'$E_{true}\, / \, GeV$')
plt.show()
plt.close()
