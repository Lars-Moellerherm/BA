import pandas as pd
import numpy as np
import scipy as sc
import h5py as h5
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score
import dask.dataframe as dd
from tqdm import tqdm


def weighted_mean_over_ID(weight, prediction):
    pred = pd.merge(prediction,weight,on=['array_event_id','run_id'])
    pred['weighted_data'] = pred['prediction']*pred['weight']
    x = pred.groupby(by=['array_event_id','run_id'])
    prediction_w_mean = x['weighted_data'].sum()/x['weight'].sum()
    prediction_w_mean = prediction_w_mean.to_frame('weighted_prediction')
    prediction_w_mean = prediction_w_mean.reset_index()
    return prediction_w_mean


def calc_scaled_width_and_length(data):
    SW = (data.width - np.mean(data.width))/sc.stats.sem(data.width)
    SL = (data.length - np.mean(data.length))/sc.stats.sem(data.length)
    SV = pd.DataFrame({'scaled_length':SL, 'scaled_width':SW, 'array_event_id':data['array_event_id'], 'run_id':data['run_id']})
    data = data.set_index(['run_id','array_event_id'])
    SV = SV.set_index(['run_id','array_event_id'])
    data = pd.concat([data,SV],axis=1).reset_index()
    return data

def calc_mean_scaled_width_and_length(X):
    X = calc_scaled_width_and_length(X)
    msl = X[['scaled_length','run_id','array_event_id']].groupby(by=['run_id','array_event_id']).mean()
    msw = X[['scaled_width','run_id','array_event_id']].groupby(by=['run_id','array_event_id']).mean()
    msl=msl.rename(columns={'scaled_length':'mean_scaled_length'})
    msw=msw.rename(columns={'scaled_width':'mean_scaled_width'})
    sl_std = X[['scaled_length','run_id','array_event_id']].groupby(by=['run_id','array_event_id']).std()
    sw_std = X[['scaled_width','run_id','array_event_id']].groupby(by=['run_id','array_event_id']).std()
    sl_std=sl_std.rename(columns={'scaled_length':'std_scaled_length'})
    sw_std=sw_std.rename(columns={'scaled_width':'std_scaled_width'})
    X=X.drop(['scaled_length','scaled_width','width','length'],axis=1).drop_duplicates().set_index(['run_id','array_event_id'])
    X = pd.concat([X,msl,msw,sl_std,sw_std],axis=1)
    return X


def plot_hist2d(predictions,truth,min_energy,max_energy,bin_edges):
    min_e = np.log10(min_energy)
    max_e = np.log10(max_energy)
    r2=r2_score(predictions,truth)
    d, bin1, bin2 = np.histogram2d(predictions, truth, bins=bin_edges)
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.pcolormesh(bin1, bin2, d, cmap='viridis', norm=LogNorm())
    #plt.grid(True,which='both')
    plt.colorbar()
    plt.plot([min_energy,max_energy],[min_energy,max_energy],color="grey", label= "richtige Schätzung")
    #plt.legend()
    ax.text(0.2, 0.95,r'$R^2$: %.2f'% r2, ha='center', va='center', transform=ax.transAxes,size='medium',bbox=dict(boxstyle="round",facecolor='grey',alpha=0.1))
    plt.ylabel('Schätzung / TeV')
    plt.xlabel('Wahrheit / TeV')
    plt.xscale('log')
    plt.yscale('log')
    #plt.xlim(max_e)
    #plt.ylim(max_e)

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
    maxi = y.min()
    f, (ax,ax2) = plt.subplots(2,1,sharex=True)

    ax2.plot(bin_df['bin'],y,'.')
    ax.plot(bin_df['bin'],y,'.')
    ax2.set_ylim(maxi-1,maxi+1)
    ax.set_ylim(-4,1)
    ax2.set_xscale('log')
    ax.set_xscale('log')
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop='off')
    ax2.xaxis.tick_bottom()


    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    ax2.set_xlabel('bin center in TeV')
    ax2.set_ylabel('r2 score')



def reading_data(diffuse,data_size1):
    # Import data in h5py
    gammas = h5.File("../data/3_gen/gammas.hdf5","r")
    # Converting to pandas
    gamma_array_df = pd.DataFrame(data=dict(gammas['array_events']))
    gamma_runs_df = pd.DataFrame(data=dict(gammas['runs']))
    gamma_telescope_df = pd.DataFrame(data=dict(gammas['telescope_events']))
    max_size = gamma_telescope_df.shape[0]
    if(data_size1 < 0):
        data_size = max_size-1
    else:
        data_size = data_size1


    #merging of array and telescope data and shuffle of proton and gamma
    gamma_merge = pd.merge(gamma_telescope_df,gamma_array_df,on=list(["array_event_id",'run_id']))
    #there are some nan in width the needed to be deleted
    gamma_merge = gamma_merge.dropna(axis=1,how='all')
    gamma_merge = gamma_merge.dropna(axis=0)
    data = gamma_merge.iloc[:data_size]
    print("Anzahl an pointlike: ",data.shape)


    if(diffuse):
        gammas_diffuse = h5.File("../data/3_gen/gammas_diffuse.hdf5","r")

        gamma_diffuse_array_df = pd.DataFrame(data=dict(gammas_diffuse['array_events']))
        gamma_diffuse_runs_df = pd.DataFrame(data=dict(gammas_diffuse['runs']))
        gamma_diffuse_telescope_df = pd.DataFrame(data=dict(gammas_diffuse['telescope_events']))

        max_size_diffuse = gamma_diffuse_telescope_df.shape[0]
        if(data_size1 < 0):
            data_size = max_size_diffuse-1
        else:
            data_size = data_size1

        gamma_diffuse_merge = pd.merge(gamma_diffuse_array_df,gamma_diffuse_telescope_df,on=list(['array_event_id','run_id']))
        gamma_diffuse_merge = gamma_diffuse_merge.dropna(axis=1,how='all')
        gamma_diffuse_merge = gamma_diffuse_merge.dropna(axis=0)
        gamma_diffuse_merge = gamma_diffuse_merge.iloc[:data_size]
        data = pd.concat([data,gamma_diffuse_merge])
        data = data.dropna(axis=1)
        print("Data of diffuse: ",gamma_diffuse_merge.shape)
        print("Data gesamt: ",data.shape)

        print("Using diffused data...")

    print("Komplette Anzahl an untersuchter Events: ",data.shape[0])
    return data;


def drop_data(data):

    droped_information = list(data)
    used = ['length','width','num_triggered_telescopes','intensity','kurtosis','skewness','total_intensity',
            'telescope_type_id','num_triggered_lst','num_triggered_mst','num_triggered_sst','array_event_id','run_id','mc_energy']
    if( droped_information.count('scaled_length')==1):
        used.append('scaled_length')
        used.append('scaled_width')
    droped_information = [e for e in droped_information if e not in used]
    droped_data = data[droped_information].copy(deep=True)
    data = data.drop(droped_information,axis=1)


    return data, droped_data


def bias(df,
        bins,
        ax_bias=None,
        prediction_key='gamma_energy_prediction',
        true_energy_key='corsika_event_header_total_energy',
        mark='.',
        color='C0',
        label='prediction'
        ):

        df['bin'] = np.digitize(df[true_energy_key], bins)
        df['rel_error'] = (df[prediction_key] - df[true_energy_key]) / df[true_energy_key]

        binned = pd.DataFrame(index=np.arange(1, len(bins)))
        binned['center'] = 0.5 * (bins[:-1] + bins[1:])
        binned['width'] = np.diff(bins)

        bias = []

        for i in tqdm(range(100)):
            grouped = df.sample(len(df), replace=True).groupby('bin')
            bias.append(grouped['rel_error'].mean())

        bias = pd.concat(bias, axis=1)

        binned['bias'] = bias.mean(axis=1)
        binned['bias_err'] = bias.std(axis=1)

        ax_bias.errorbar(
            binned['center'],
            binned['bias'],
            xerr=0.5 * binned['width'],
            yerr=binned['bias_err'],
            label=label,
            linestyle='',
            marker = mark,
            color=color
        )


def resolution(df,
        bins,
        ax_res=None,
        prediction_key='gamma_energy_prediction',
        true_energy_key='corsika_event_header_total_energy',
        mark='.',
        color='C1',
        label='Resolution',
        std=False
        ):

        df['bin'] = np.digitize(df[true_energy_key], bins)
        df['rel_error'] = (df[prediction_key] - df[true_energy_key]) / df[true_energy_key]

        binned = pd.DataFrame(index=np.arange(1, len(bins)))
        binned['center'] = 0.5 * (bins[:-1] + bins[1:])
        binned['width'] = np.diff(bins)

        resolution_quantiles = []
        resolution_stds = []

        for i in tqdm(range(100)):
            grouped = df.sample(len(df), replace=True).groupby('bin')
            lower_sigma = grouped['rel_error'].agg(lambda s: np.percentile(s, 15.87))
            upper_sigma = grouped['rel_error'].agg(lambda s: np.percentile(s, 84.13))
            resolution_quantiles.append(0.5 * (upper_sigma - lower_sigma))
            resolution_stds.append(grouped.rel_error.std())

        resolution_quantiles = pd.concat(resolution_quantiles, axis=1)
        resolution_stds = pd.concat(resolution_stds, axis=1)

        binned['resolution_quantiles'] = resolution_quantiles.mean(axis=1)
        binned['resolution_quantiles_err'] = resolution_quantiles.std(axis=1)
        binned['resolution'] = resolution_stds.mean(axis=1)
        binned['resolution_err'] = resolution_stds.std(axis=1)

        if std is False:
            ax_res.errorbar(
                binned['center'],
                binned['resolution_quantiles'],
                xerr=0.5 * binned['width'],
                yerr=binned['resolution_quantiles_err'],
                label=label,
                linestyle='',
                marker=mark,
                color=color,
            )
        else:
            ax_res.errorbar(
                binned['center'],
                binned['resolution'],
                yerr=binned['resolution_err'],
                xerr=0.5 * binned['width'],
                label=label,
                linestyle='',
                marker=mark,
                color=color,
            )


def translate(names,kind):
    name = np.asarray(names)
    if(kind==1):
        name[name=='num_triggered_sst']='Anz_ausgelöster_SST'
        name[name=='num_triggered_mst']='Anz_ausgelöster_MST'
        name[name=='num_triggered_lst']='Anz_ausgelöster_LST'
        name[name=='total_intensity']='totale_Intensität'
        name[name=='num_triggered_telescopes']='Anz_ausgelöster_Teleskope'
        name[name=='telescope_type_id']='Teleskopart'
        name[name=='intensity']='Intensität'
        name[name=='scaled_length']='skalierte_Länge'
        name[name=='scaled_width']='skalierte_Weite'
        name[name=='length']='Länge'
        name[name=='width']='Weite'
        name[name=='skewness']='Schiefe'
        name[name=='kurtosis']='Wölbung'
    if(kind==2):
        name[name=='min_mst_pred']='min_MST_Schätzung'
        name[name=='max_mst_pred']='max_MST_Schätzung'
        name[name=='mean_mst_pred']='Mittelwert_MST_Schätzung'
        name[name=='total_intensity']='totale_Intensität'
        name[name=='num_triggered_sst']='Anz_ausgelöster_SST'
        name[name=='mean_sst_pred']='Mittelwert_SST_Schätzung'
        name[name=='mean_prediction']='Mittelwert_Schätzung'
        name[name=='min_sst_pred']='min_SST_Schätzung'
        name[name=='std_sst_pred']='std_SST_Schätzung'
        name[name=='max_sst_pred']='max_SST_Schätzung'
        name[name=='num_triggered_telescopes']='Anz_ausgelöster_Teleskope'
        name[name=='mean_lst_pred']='Mittelwert_LST_Schätzung'
        name[name=='min_lst_pred']='min_LST_Schätzung'
        name[name=='std_mst_pred']='std_MST_Schätzung'
        name[name=='std_scaled_length']='std_skalierte_Länge'
        name[name=='std_scaled_width']='std_skalierte_Weite'
        name[name=='num_triggered_mst']='Anz_ausgelöster_MST'
        name[name=='std_lst_pred']='std_LST_Schätzung'
        name[name=='mean_scaled_length']='Mittelwert_skalierte_Länge'
        name[name=='max_lst_pred']='max_LST_Schätzung'
        name[name=='num_triggered_lst']='Anz_ausgelöster_LST'
        name[name=='mean_scaled_width']='Mittelwert_skalierte_Weite'
    return name
