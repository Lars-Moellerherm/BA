import numpy as np
import matplotlib.pyplot as plt
import functions as func


min_energy = 0.003
max_energy = 340
bin_edge = np.logspace(np.log10(min_energy),np.log10(max_energy),35)

  ############PLotting of encapsulated_RF.py################

#reading data
predictions, truth = np.genfromtxt("../good_data/encaps_pred_data.txt", unpack=True)
prediction_w_mean, truth_w_mean = np.genfromtxt("../good_data/encaps_pred_w_mean_data.txt", unpack=True)
prediction_w2_mean, truth_w2_mean = np.genfromtxt("../good_data/encaps_pred_w2_mean_data.txt",unpack=True)
predictions_encaps, truth_encaps = np.genfromtxt("../good_data/encaps_encaps_pred_data.txt", unpack=True)
print('finished with reading data of encapsulated_RF.py ... \n')

plt.subplot(221)
r2_1 = func.plot_hist2d(predictions,truth,min_energy,max_energy,bin_edge)
plt.title("RF(with MSV)(R2score: %.2f)" % r2_1)

    #weighted mean (ist_to_core)
plt.subplot(222)
r2_4 = func.plot_hist2d(prediction_w2_mean,truth_w2_mean,min_energy,max_energy,bin_edge)
plt.title("(MSV) w mean(dist_to_core)(%.2f)" % r2_4)

    #weighted mean (intensity)
plt.subplot(223)
r2_2 = func.plot_hist2d(prediction_w_mean,truth_w_mean,min_energy,max_energy,bin_edge)
plt.title("(MSV) w mean(inty)(%0.2f)" % r2_2 )

    #plots for encapsulated RF
plt.subplot(224)
r2_3 = func.plot_hist2d(predictions_encaps,truth_encaps,min_energy,max_energy,bin_edge)
plt.title("(MSV,w inty) encap RFr(%.2f)" % r2_3)
plt.subplots_adjust(wspace=0.45,hspace=0.45)
#plt.show()
plt.savefig('RF_Regression_MSV_all.jpg')
plt.close()


########################R2 score per bin ###########################
plt.subplot(221)
func.plot_R2_per_bin(predictions,truth,bin_edge)
plt.title("RF(with MSV)")

    #weighted mean (ist_to_core)
plt.subplot(222)
func.plot_R2_per_bin(prediction_w2_mean,truth_w2_mean,bin_edge)
plt.title("(MSV) w mean(dist_to_core)")

    #weighted mean (intensity)
plt.subplot(223)
func.plot_R2_per_bin(prediction_w_mean,truth_w_mean,bin_edge)
plt.title("(MSV) w mean(inty)")

    #plots for encapsulated RF
plt.subplot(224)
func.plot_R2_per_bin(predictions_encaps,truth_encaps,bin_edge)
plt.title("(MSV)encap")
plt.subplots_adjust(wspace=0.45,hspace=0.5)
#plt.show()
plt.savefig('RF_Regression_MSV_R2_per_bin_all.jpg')
plt.close()


#what to add:
##-an title for the whole subplots
##-cut the y-axis between the first and the other points to see more of the first values
