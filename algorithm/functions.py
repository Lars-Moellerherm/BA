import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def weighted_mean_over_ID(predictions, ID, weight, truth):

    unique_ID = ID.unique()
    pred_ID = pd.DataFrame({'predicted_energy':predictions, 'array_event_id':ID})
    truth_ID = pd.DataFrame({'mc_energy':truth, 'array_event_id':ID})
    truth_unique = pd.Series([], name='mc_energy')

        # With weighted mean  INTENSITY
    weight_ID = pd.DataFrame({'weights': weight, 'array_event_id': ID})
    prediction_w_mean = pd.DataFrame({'predicted_energy': [], 'array_event_id': []})

    for i in unique_ID:
        #INTENSITY
        x = weight_ID.weights[weight_ID.array_event_id == i]
        pred_w_mean = np.average(pred_ID.predicted_energy[pred_ID.array_event_id == i], weights=x)
        pred_w_mean = {'predicted_energy' : pd.Series([pred_w_mean]), 'array_event_id' : pd.Series([i])}
        pred_w_mean = pd.DataFrame(pred_w_mean)
        prediction_w_mean = pd.concat([prediction_w_mean,pred_w_mean], ignore_index=True)

        #make the truth unique
        y = truth_ID.mc_energy[truth_ID.array_event_id == i].iloc[0]
        y = pd.Series(y, name='mc_energy')
        truth_unique = pd.concat([truth_unique, y], ignore_index=True)


    return prediction_w_mean, truth_unique


def calc_mean_scaled_width_and_length(data):

    ID = data['array_event_id']
    unique_ID = ID.unique()
    mean_scaled_width = pd.DataFrame({'mean_scaled_w':[],'array_event_id':[]})
    mean_scaled_length = pd.DataFrame({'mean_scaled_l':[],'array_event_id':[]})
    SW = (data.width - np.mean(data.width))/sc.stats.sem(data.width)
    scaled_width = pd.DataFrame({'scaled_w':SW, 'array_event_id':ID})
    SL = (data.length - np.mean(data.length))/sc.stats.sem(data.length)
    scaled_length = pd.DataFrame({'scaled_l':SL, 'array_event_id':ID})

    for i in unique_ID:
        ntels = data.num_triggered_telescopes[data.array_event_id == i].iloc[0]
        MSW = np.sum(scaled_width.scaled_w[scaled_width.array_event_id == i])
        MSW = MSW/np.sqrt(ntels)
        MSW = pd.Series(MSW)
        MSW = pd.DataFrame({'mean_scaled_w':MSW,'array_event_id':i})
        mean_scaled_width = pd.concat([mean_scaled_width,MSW], ignore_index=True)

        MSL = np.sum(scaled_length.scaled_l[scaled_length.array_event_id == i])
        MSL = MSL/np.sqrt(ntels)
        MSL = pd.Series(MSL)
        MSL = pd.DataFrame({'mean_scaled_l':MSL,'array_event_id':i})
        mean_scaled_length = pd.concat([mean_scaled_length,MSL], ignore_index=True)


    data = pd.merge(data,mean_scaled_width, on='array_event_id')
    data = pd.merge(data, mean_scaled_length, on='array_event_id')

    return data


def plot_hist2d(predictions,truth,min_energy,max_energy):
    min_e = np.log10(min_energy)
    max_e = np.log10(max_energy)
    bin_edges = np.logspace(min_e,max_e,60)
    r2=r2_score(predictions,truth.values)
    plt.hist2d(predictions, truth.values, bins=bin_edges, cmap="viridis", cmin=1)
    plt.grid(True,which='both')
    plt.colorbar()
    plt.plot([0.003,330],[0.003,330],color="grey", label= "correct prediction")
    plt.legend()
    plt.xlabel('Predicted value / TeV')
    plt.ylabel('truth value / TeV')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(max_e)
    plt.ylim(max_e)

    return r2

def plot_error(predictions,truth):
    error = (predictions-truth.values)**2
    bin_edges = np.logspace(np.log10(0.0001),np.log10(5),60)
    plt.hist(error,bins=bin_edges)
    plt.xlabel(r'squared errors in $TeV^2$')
    plt.ylabel('counts')
    plt.xscale('log')
