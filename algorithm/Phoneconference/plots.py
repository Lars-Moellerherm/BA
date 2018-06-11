import numpy as np
import matplotlib.pyplot as plt
import functions as func
import pandas as pd
from sklearn.metrics import r2_score
from brokenaxes import brokenaxes

min_energy = 0.003
max_energy = 340
bin_edge = np.logspace(np.log10(min_energy),np.log10(max_energy),35)

  ############PLotting of encapsulated_RF.py################

#reading data
predictions, truth = np.genfromtxt("../good_data/encaps_pred_data.txt", unpack=True)
prediction_w_mean, truth_w_mean = np.genfromtxt("../good_data/encaps_pred_w_mean_data.txt", unpack=True)
predictions_encaps, truth_encaps = np.genfromtxt("../good_data/encaps_encaps_pred_data.txt", unpack=True)
print('finished with reading data of encapsulated_RF.py ... \n')


r2_1 = func.plot_hist2d(predictions,truth,min_energy,max_energy,bin_edge)
plt.savefig("RF_MSV.jpg")
plt.close()


    #weighted mean (intensity)
r2_2 = func.plot_hist2d(prediction_w_mean,truth_w_mean,min_energy,max_energy,bin_edge)
plt.savefig("RF_MSV_wI_mean.jpg")
plt.close()

    #plots for encapsulated RF
r2_3 = func.plot_hist2d(predictions_encaps,truth_encaps,min_energy,max_energy,bin_edge)
plt.savefig('RF_MSV_encaps.jpg')
plt.close()


########################R2 score per bin ###########################
func.plot_R2_per_bin(predictions,truth,bin_edge)
plt.tight_layout()
plt.savefig("RF_MSV_R2_per_bin.jpg")
plt.close()

    #weighted mean (intensity)
func.plot_R2_per_bin(prediction_w_mean,truth_w_mean,bin_edge)
plt.tight_layout()
plt.savefig("RF_MSV_wI_mean_R2_per_bin.jpg")
plt.close()

    #plots for encapsulated RF
func.plot_R2_per_bin(predictions_encaps,truth_encaps,bin_edge)
plt.tight_layout()
plt.savefig('RF_MSV_encaps_R2_per_bin.jpg')
plt.close()

############################ RandomForestRegressor.py ###########################################

#reading data

prediction, truth = np.genfromtxt("../good_data/RFr_pred_data.txt",unpack=True)
prediction_mean, truth_mean = np.genfromtxt("../good_data/RFr_pred_mean_data.txt",unpack=True)
prediction_wI, truth_wI = np.genfromtxt("../good_data/RFr_pred_wI_mean_data.txt",unpack=True)
prediction_wS, truth_wS = np.genfromtxt("../good_data/RFr_pred_wS_mean_data.txt",unpack=True)
prediction_wT, truth_wT = np.genfromtxt("../good_data/RFr_pred_wT_mean_data.txt",unpack=True)

print("finished with reading data of RandomForestRegressor.py ")

r2_1 = func.plot_hist2d(prediction,truth,min_energy,max_energy,bin_edge)
plt.savefig("RF.jpg")
plt.close()

r2_1 = func.plot_hist2d(prediction_mean,truth_mean,min_energy,max_energy,bin_edge)
plt.savefig("RF_mean.jpg")
plt.close()

r2_1 = func.plot_hist2d(prediction_wI,truth_wI,min_energy,max_energy,bin_edge)
plt.savefig("RF_wI.jpg")
plt.close()

r2_1 = func.plot_hist2d(prediction_wS,truth_wS,min_energy,max_energy,bin_edge)
plt.savefig("RF_wS.jpg")
plt.close()

r2_1 = func.plot_hist2d(prediction_wT,truth_wT,min_energy,max_energy,bin_edge)
plt.savefig("RF_wT.jpg")
plt.close()

############### R2 per bin plot ################

func.plot_R2_per_bin(prediction,truth,bin_edge)
plt.tight_layout()
plt.savefig("RF_R2_per_bin.jpg")
plt.close()

func.plot_R2_per_bin(prediction_mean,truth_mean,bin_edge)
plt.tight_layout()
plt.savefig("RF_mean_R2_per_bin.jpg")
plt.close()

func.plot_R2_per_bin(prediction_wI,truth_wI,bin_edge)
plt.tight_layout()
plt.savefig("RF_wI_R2_per_bin.jpg")
plt.close()

func.plot_R2_per_bin(prediction_wS,truth_wS,bin_edge)
plt.tight_layout()
plt.savefig("RF_wS_R2_per_bin.jpg")
plt.close()

func.plot_R2_per_bin(prediction_wT,truth_wT,bin_edge)
plt.tight_layout()
plt.savefig("RF_wT_R2_per_bin.jpg")
plt.close()


#what to add:+
# change axes bei R2 per bin plots
