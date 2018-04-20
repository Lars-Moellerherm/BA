import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def weighted_mean_over_ID(predictions, weight, data):
    ID = data['array_event_id']
    truth = data['mc_energy']
    pred = pd.DataFrame({'predicted_energy':predictions, 'array_event_id':ID, 'mc_energy':
                        truth, 'weight': weight})
    pred['weighted_data'] = pred['predicted_energy']*pred['weight']
    x = pred.groupby('array_event_id')
    prediction_w_mean = x['weighted_data'].sum()/x['weight'].sum()
    predict = pd.DataFrame({'predicted_energy': prediction_w_mean,
                            'array_event_id': prediction_w_mean.index})
    data = pd.merge(data, predict, on='array_event_id')

    return data


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


def mean_over_ID(predictions, ID, truth):

    pred = pd.DataFrame({'predicted_energy':predictions, 'array_event_id':ID, 'mc_energy':
                        truth})
    prediction_mean = pred.groupby('array_event_id').mean()
    predict = pd.DataFrame({'predicted_energy': prediction_mean.loc[:,'predicted_energy'],
                            'array_event_id': prediction_mean.index})
    return predict, prediction_mean['mc_energy']
