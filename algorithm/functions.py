import pandas as pd
import numpy as np
import scipy as sc
import h5py as h5
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
    pred = pd.DataFrame({'predicted_energy':predictions})
    prediction_mean = pred.groupby(level=list(['array_event_id','run_id'])).mean()
    prediction_mean = prediction_mean.reset_index()
    data = data.reset_index()
    data2 = pd.merge(data, prediction_mean, on=list(['run_id','array_event_id']))
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


def plot_std_der_bins(predictions,truth,bins):
    df = pd.DataFrame({'predictions':predictions,'truth':truth})
    df['cut'] = pd.cut(x=df['truth'],bins=bins)
    standard = df.groupby('cut').std()
    standard['mean'] = df.groupby('cut').mean()['truth']
    plt.plot(standard['mean'],standard['predictions'],'x')
    plt.xlabel(r'$\mu$ in jedem Bin / TeV')
    plt.ylabel(r'$\sigma$ für jeden Bin')
    plt.xscale('log')


def plot_R2_per_bin(prediction, truth, bins):
    df = pd.DataFrame({'prediction':prediction,'truth':truth})
    label = np.arange(1,bins.size)
    left_edge_df = pd.DataFrame(data = bins[:-1], index = label,columns=['bin'])
    right_edge_df = pd.DataFrame(data = bins[1:], index = label, columns=['bin'])
    bin_df = (left_edge_df+right_edge_df)/2
    df['cut'] = pd.cut(x=df['truth'],bins=bins,labels=label)
    n=0
    y = np.ones(bins.size-1)
    grouped = df.groupby('cut')
    for i in label:
        if(df['cut'].isin([i]).sum()):
            group = grouped.get_group(i)
            y[n] = r2_score(group['prediction'],group['truth'])
        else:
            y[n]=0.0
        n+=1
    plt.plot(bin_df['bin'],y,'.')
    plt.xscale('log')
    plt.xlabel('bin center in TeV')
    plt.ylabel('r2 score')


def reading_data(diffuse,data_size1):
    # Import data in h5py
    gammas = h5.File("../data/3_gen/gammas.hdf5","r")

    # Converting to pandas
    gamma_array_df = pd.DataFrame(data=dict(gammas['array_events']))
    gamma_runs_df = pd.DataFrame(data=dict(gammas['runs']))
    gamma_telescope_df = pd.DataFrame(data=dict(gammas['telescope_events']))
    max_size = gamma_array_df.shape[0]
    if(data_size1 >= max_size):
        data_size = max_size-1
    else:
        data_size = data_size1
    gamma_array_df = gamma_array_df.iloc[:data_size]
    gamma_runs_df = gamma_runs_df.iloc[:data_size]
    gamma_telescope_df = gamma_telescope_df.iloc[:data_size]


    #merging of array and telescope data and shuffle of proton and gamma
    gamma_merge = pd.merge(gamma_array_df,gamma_telescope_df,on=list(["array_event_id",'run_id']))
    gamma_merge = gamma_merge.set_index(['run_id','array_event_id'])
    #there are some nan in width the needed to be deleted
    gamma_merge = gamma_merge.dropna(axis=0)
    data = gamma_merge


    if(diffuse):
        gammas_diffuse = h5.File("../data/3_gen/gammas_diffuse.hdf5","r")

        gamma_diffuse_array_df = pd.DataFrame(data=dict(gammas_diffuse['array_events']))
        max_size_diffuse = gamma_diffuse_array_df.shape[0]
        if(data_size1-1 >= max_size_diffuse):
            data_size = max_size_diffuse-1
        else:
            data_size = data_size1

        gamma_diffuse_array_df = gamma_diffuse_array_df.iloc[:data_size]
        gamma_diffuse_runs_df = pd.DataFrame(data=dict(gammas_diffuse['runs']))
        gamma_diffuse_runs_df = gamma_diffuse_runs_df.iloc[:data_size]
        gamma_diffuse_telescope_df = pd.DataFrame(data=dict(gammas_diffuse['telescope_events']))
        gamma_diffuse_telescope_df = gamma_diffuse_telescope_df.iloc[:data_size]
        gamma_diffuse_merge = pd.merge(gamma_diffuse_array_df,gamma_diffuse_telescope_df,on=list(['array_event_id','run_id']))
        gamma_diffuse_merge = gamma_diffuse_merge.set_index(['run_id','array_event_id'])
        gamma_diffuse_merge = gamma_diffuse_merge.dropna(axis=0)
        gamma_diffuse_merge = gamma_diffuse_merge.reset_index()
        gamma_merge = gamma_merge.reset_index()
        data = pd.concat([gamma_merge,gamma_diffuse_merge])
        data = data.set_index(['run_id','array_event_id'])
        data = data.dropna(axis=1)
        print("Using diffused data...")

    return data;


def drop_data(data):

    droped_information = list(['psi','phi','telescope_type_name','x','y','telescope_id','pointing_altitude',
                                'camera_name','camera_id','pointing_azimuth','r','distance_to_core',
                                'mc_az','mc_alt','mc_core_x','mc_core_y','mc_energy','mc_corsika_primary_id',
                                'mc_height_first_interaction','h_max_prediction','alt_prediction','az_prediction',
                                'core_x_prediction','core_y_prediction'])
    droped_data = data[droped_information].copy(deep=True)
    data = data.drop(droped_information,axis=1)


    return data, droped_data
