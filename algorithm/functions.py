import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score

def weighted_mean_over_ID(weight, data):
    weight2 = weight.copy(deep=True)
    predictions = data['predictions'].copy(deep=True)
    truth = data['mc_energy'].copy(deep=True)
    pred = pd.DataFrame({'predicted_energy':predictions, 'mc_energy':
                        truth, 'weight': weight2})
    pred['weighted_data'] = pred['predicted_energy']*pred['weight']
    x = pred.groupby(level=list(['array_event_id','run_id']))
    prediction_w_mean = x['weighted_data'].sum()/x['weight'].sum()
    prediction_w_mean = prediction_w_mean.to_frame('predicted_energy')
    prediction_w_mean = prediction_w_mean.reset_index()
    data = data.reset_index()
    data2 = pd.merge(data,prediction_w_mean, on=list(['array_event_id','run_id']))
    data2 = data2.set_index(list(['run_id','array_event_id']))
    return data2


def calc_mean_scaled_width_and_length(data):
    SW = (data.width - np.mean(data.width))/sc.stats.sem(data.width)
    SL = (data.length - np.mean(data.length))/sc.stats.sem(data.length)
    SV = pd.DataFrame({'scaled_length':SL, 'scaled_width':SW})
    MSV = SV.groupby(level=list(['array_event_id','run_id'])).mean()
    MSV = MSV.reset_index()
    data = data.reset_index()
    data2 = pd.merge(data,MSV,on=['array_event_id','run_id'])
    data2 = data2.set_index(['run_id','array_event_id'])
    return data2


def plot_hist2d(predictions,truth,min_energy,max_energy,bin_edges):
    min_e = np.log10(min_energy)
    max_e = np.log10(max_energy)
    r2=r2_score(predictions,truth)
    d, bin1, bin2 = np.histogram2d(predictions, truth, bins=bin_edges)
    plt.pcolormesh(bin1, bin2, d, cmap='viridis', norm=LogNorm())
    #plt.grid(True,which='both')
    plt.colorbar()
    plt.plot([min_energy,max_energy],[min_energy,max_energy],color="grey", label= "correct prediction")
    #plt.legend()
    plt.ylabel('Predicted value / TeV')
    plt.xlabel('truth value / TeV')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(max_e)
    plt.ylim(max_e)

    return r2


def plot_error(predictions,truth):
    error = (predictions-truth)**2
    bin_edges = np.logspace(np.log10(0.0001),np.log10(5),60)
    plt.hist(error,bins=bin_edges)
    plt.xlabel(r'squared errors in $TeV^2$')
    plt.ylabel('counts')
    plt.xscale('log')


def mean_over_ID(data):
    truth = data['mc_energy'].copy(deep=True)
    predictions = data['predictions'].copy(deep=True)
    pred = pd.DataFrame({'predicted_energy':predictions, 'mc_energy':
                        truth})
    prediction_mean = pred.groupby(level=list(['array_event_id','run_id'])).mean()
    prediction_mean = prediction_mean.reset_index()
    data = data.reset_index()
    data2 = pd.merge(data, predict, on=list(['run_id','array_event_id']))
    data2 = data2.set_index(['run_id','array_event_id'])

    return data2


def plot_rel_error(predictions, truth):
    rel_error = (predictions-truth)/truth
    rel_error = abs(rel_error)
    bin_edges = np.logspace(np.log10(0.0001),np.log10(3),50)
    plt.hist(rel_error,bins=bin_edges)
    plt.xlabel('relative error of prediction')
    plt.ylabel('counts')
    plt.xscale('log')

    return rel_error.mean(), sc.stats.sem(rel_error)


def plot_trueDIVpred(predictions, truth):
    div = truth/predictions
    bin_edges = np.logspace(np.log10(min(div)),np.log10(max(div)),50)
    plt.hist(div,bins=bin_edges)
    plt.xlabel('truth/prediction')
    plt.ylabel('counts')
    plt.xscale('log')

    return div.mean(), sc.stats.sem(div)


def plot_std_der_bins(predictions,bins):
    df = pd.DataFrame({'predictions':predictions})
    df['cut'] = pd.cut(x=predictions,bins=bins)
    standard = df.groupby('cut').std()
    standard['mean'] = df.groupby('cut').mean()
    plt.plot(np.log10(standard['mean']),standard['predictions'],'x')
    plt.xlabel(r'$\mu$ in jedem Bin / TeV')
    plt.ylabel(r'$\sigma$ für jeden Bin')
    plt.xscale('log')
