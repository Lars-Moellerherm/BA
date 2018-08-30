import numpy as np
import matplotlib.pyplot as plt
import functions as func
import pandas as pd
from sklearn.metrics import r2_score
from scipy import stats


min_energy = 0.003
max_energy = 340
bin_edge = np.logspace(np.log10(min_energy),np.log10(max_energy),300)
bin_edge2 = np.logspace(np.log10(0.01),np.log10(max_energy),20)

  ############PLotting of encapsulated_RF.py################

#reading data
predictions, truth = np.genfromtxt("../good_data/encaps_pred_data.txt", unpack=True)
prediction_w_mean, truth_w_mean = np.genfromtxt("../good_data/encaps_pred_mean_data.txt", unpack=True)
prediction_mean, truth_mean = np.genfromtxt("../good_data/encaps_pred_median_data.txt", unpack=True)
predictions_encaps, truth_encaps = np.genfromtxt("../good_data/encaps_encaps_pred_data.txt", unpack=True)
print('finished with reading data of encapsulated_RF.py ... \n')


r2_1 = func.plot_hist2d(predictions,truth,min_energy,max_energy,bin_edge)
plt.savefig("RF.pdf")
plt.close()


    #weighted mean (intensity)
r2_2 = func.plot_hist2d(prediction_w_mean,truth_w_mean,min_energy,max_energy,bin_edge)
plt.savefig("RF_mean.pdf")
plt.close()

    # just the mean
r2_4 = func.plot_hist2d(prediction_mean,truth_mean,min_energy,max_energy,bin_edge)
plt.savefig("RF_median.pdf")
plt.close()

    #plots for encapsulated RF
r2_3 = func.plot_hist2d(predictions_encaps,truth_encaps,min_energy,max_energy,bin_edge)
plt.savefig('RF_encaps.pdf')
plt.close()


########################R2 score per bin ###########################
    #### just MSV
#bins = np.logspace(np.log10(min_energy),np.log10(max_energy),35)
#df1 = pd.DataFrame({'prediction':predictions,'truth':truth})
#label = np.arange(1,bins.size)
#left_edge_df = pd.DataFrame(data = bins[:-1], index = label,columns=['bin'])
#right_edge_df = pd.DataFrame(data = bins[1:], index = label, columns=['bin'])
#bin_df = (left_edge_df+right_edge_df)/2
#df1['cut'] = pd.cut(x=df1['truth'],bins=bins,labels=label)
#n=0
#y1 = np.ones(bins.size-1)
#grouped1 = df1.groupby('cut')
#for i in label:
#    if(df1['cut'].isin([i]).sum()):
#        group1 = grouped1.get_group(i)
#        y1[n] = r2_score(group1['prediction'],group1['truth'])
#    else:
#        y1[n]=0.0
#    n+=1
#maxi1 = y1.min()
#
#    ##### w mean Intensity
#df2 = pd.DataFrame({'prediction':prediction_w_mean,'truth':truth_w_mean})
#df2['cut'] = pd.cut(x=df2['truth'],bins=bins,labels=label)
#n=0
#y2 = np.ones(bins.size-1)
#grouped2 = df2.groupby('cut')
#for i in label:
#    if(df2['cut'].isin([i]).sum()):
#        group2 = grouped2.get_group(i)
#        y2[n] = r2_score(group2['prediction'],group2['truth'])
#    else:
#        y2[n]=0.0
#    n+=1
#maxi2 = y2.min()
#
#    #### encaps RF
#df3 = pd.DataFrame({'prediction':predictions_encaps,'truth':truth_encaps})
#df3['cut'] = pd.cut(x=df3['truth'],bins=bins,labels=label)
#n=0
#y3 = np.ones(bins.size-1)
#grouped3 = df3.groupby('cut')
#for i in label:
#    if(df3['cut'].isin([i]).sum()):
#        group3 = grouped3.get_group(i)
#        y3[n] = r2_score(group3['prediction'],group3['truth'])
#    else:
#        y3[n]=0.0
#    n+=1
#maxi3 = y3.min()
#maxi=np.array([maxi1,maxi2,maxi3])
#
#f, (ax,ax2) = plt.subplots(2,1,sharex=True)
#
#ax2.plot(bin_df['bin'],y1,'b.',label="RF with MSV")
#ax.plot(bin_df['bin'],y1,'b.')
#ax2.plot(bin_df['bin'],y2,'k.',label="weigted mean (Intensity)")
#ax.plot(bin_df['bin'],y2,'k.')
#ax2.plot(bin_df['bin'],y3,'r.',label="encapsulated RF")
#ax.plot(bin_df['bin'],y3,'r.')
#ax2.set_ylim(np.min(maxi)-10,np.max(maxi)+10)
#ax.set_ylim(-7,1)
#ax2.set_xscale('log')
#ax.set_xscale('log')
#ax.spines['bottom'].set_visible(False)
#ax2.spines['top'].set_visible(False)
#ax.xaxis.tick_top()
#ax.tick_params(labeltop='off')
#ax2.xaxis.tick_bottom()
#
#
#d = .015  # how big to make the diagonal lines in axes coordinates
## arguments to pass to plot, just so we don't keep repeating them
#kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
#ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
#ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
#
#kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
#ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
#ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
#ax2.set_xlabel('bin center in TeV')
#ax2.set_ylabel('r2 score')
#plt.legend(loc='lower right')
#plt.tight_layout()
#plt.savefig('RF_MSV_R2.pdf')
#plt.close()
#
#

############ std and mean of rel error in every bin #############################

def percentilesigma(y):
    return np.percentile(y,q=68.3)


rel_error = (predictions-truth)/truth
rel_error_w_mean = (prediction_w_mean-truth_w_mean)/truth_w_mean
rel_error_mean = (prediction_mean-truth_mean)/truth_mean
rel_error_encaps = (predictions_encaps-truth_encaps)/truth_encaps
perc, bins_p, binnumber_p = stats.binned_statistic(truth,rel_error,statistic=percentilesigma,bins=bin_edge2)
perc_w_mean, bins_p_w_mean, binnumber_p_w_mean = stats.binned_statistic(truth_w_mean,rel_error_w_mean,statistic=percentilesigma,bins=bin_edge2)
perc_mean, bins_p_mean, binnumber_p_mean = stats.binned_statistic(truth_mean,rel_error_mean,statistic=percentilesigma,bins=bin_edge2)
perc_encaps, bins_p_encaps, binnumber_p_encaps = stats.binned_statistic(truth_encaps,rel_error_encaps,statistic=percentilesigma,bins=bin_edge2)


bin_p = (bins_p[:-1]+bins_p[1:])/2
plt.plot(bin_p,perc,'rx',label='RF with MSV')
plt.plot(bin_p,perc_w_mean,'bx',label='with mean')
plt.plot(bin_p,perc_mean,'kx',label='with median')
plt.plot(bin_p,perc_encaps,'gx',label='encapsulated RF')
plt.legend(loc='best')
plt.xscale('log')
plt.xlabel("Energy / TeV")
plt.tight_layout()
plt.savefig("RF_rel_std.pdf")
plt.close()


mean, bins_m, binnumber_m = stats.binned_statistic(truth,rel_error,statistic='mean',bins=bin_edge2)
mean_w_mean, bins_m_w_mean, binnumber_m_w_mean = stats.binned_statistic(truth_w_mean,rel_error_w_mean,statistic='mean',bins=bin_edge2)
mean_mean, bins_m_mean, binnumber_m_mean = stats.binned_statistic(truth_mean,rel_error_mean,statistic='mean',bins=bin_edge2)
mean_encaps, bins_m_encaps, binnumber_m_encaps = stats.binned_statistic(truth_encaps,rel_error_encaps,statistic='mean',bins=bin_edge2)

bin_m = (bins_m[:-1]+bins_m[1:])/2
plt.plot(bin_m,mean,'r.',label='RF with MSV')
plt.plot(bin_m,mean_w_mean,'b.',label='with mean')
plt.plot(bin_m,mean_mean,'k.',label="with median")
plt.plot(bin_m,mean_encaps,'g.',label='encapsulated RF')
plt.legend(loc='best')
plt.xscale('log')
plt.xlabel("Energy / TeV")
plt.tight_layout()
plt.savefig("RF_rel_mean.pdf")
plt.close()
############################ RandomForestRegressor.py ###########################################

#reading data

#prediction, truth = np.genfromtxt("../good_data/RFr_pred_data.txt",unpack=True)
#prediction_mean, truth_mean = np.genfromtxt("../good_data/RFr_pred_mean_data.txt",unpack=True)
#prediction_wI, truth_wI = np.genfromtxt("../good_data/RFr_pred_wI_mean_data.txt",unpack=True)
#prediction_wS, truth_wS = np.genfromtxt("../good_data/RFr_pred_wS_mean_data.txt",unpack=True)
#prediction_wT, truth_wT = np.genfromtxt("../good_data/RFr_pred_wT_mean_data.txt",unpack=True)
#
#print("finished with reading data of RandomForestRegressor.py ")
#
#r2_1 = func.plot_hist2d(prediction,truth,min_energy,max_energy,bin_edge)
#plt.savefig("RF.pdf")
#plt.close()
#
#r2_1 = func.plot_hist2d(prediction_mean,truth_mean,min_energy,max_energy,bin_edge)
#plt.savefig("RF_mean.pdf")
#plt.close()
#
#r2_1 = func.plot_hist2d(prediction_wI,truth_wI,min_energy,max_energy,bin_edge)
#plt.savefig("RF_wI.pdf")
#plt.close()
#
#r2_1 = func.plot_hist2d(prediction_wS,truth_wS,min_energy,max_energy,bin_edge)
#plt.savefig("RF_wS.pdf")
#plt.close()
#
#r2_1 = func.plot_hist2d(prediction_wT,truth_wT,min_energy,max_energy,bin_edge)
#plt.savefig("RF_wT.pdf")
#plt.close()
#
################ R2 per bin plot ################
#
#func.plot_R2_per_bin(prediction,truth,bin_edge)
#plt.tight_layout()
#plt.savefig("RF_R2_per_bin.pdf")
#plt.close()
#
#func.plot_R2_per_bin(prediction_mean,truth_mean,bin_edge)
#plt.tight_layout()
#plt.savefig("RF_mean_R2_per_bin.pdf")
#plt.close()
#
#func.plot_R2_per_bin(prediction_wI,truth_wI,bin_edge)
#plt.tight_layout()
#plt.savefig("RF_wI_R2_per_bin.pdf")
#plt.close()
#
#func.plot_R2_per_bin(prediction_wS,truth_wS,bin_edge)
#plt.tight_layout()
#plt.savefig("RF_wS_R2_per_bin.pdf")
#plt.close()
#
#func.plot_R2_per_bin(prediction_wT,truth_wT,bin_edge)
#plt.tight_layout()
#plt.savefig("RF_wT_R2_per_bin.pdf")
#plt.close()
#
#
##what to add:+
## change axes bei R2 per bin plots
#
