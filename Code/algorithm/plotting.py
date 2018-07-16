import matplotlib.pyplot as plt
import functions as func
import numpy as np
import pandas as pd
import argparse
from scipy import stats


def plot():

    min_energy = 0.003
    max_energy = 340
    bin_edge = np.logspace(np.log10(min_energy),np.log10(max_energy),300)


      ############PLotting of encapsulated_RF.py################
    #reading data
    predictions, truth = np.genfromtxt("good_data/encaps_pred_data.txt", unpack=True)
    prediction_mean, truth_mean = np.genfromtxt("good_data/encaps_pred_mean_data.txt", unpack=True)
    prediction_median, truth_median = np.genfromtxt("good_data/encaps_pred_median_data.txt",unpack=True)
    prediction_wI, truth_wI = np.genfromtxt("good_data/encaps_pred_wI_data.txt",unpack=True)
    prediction_wS, truth_wS = np.genfromtxt("good_data/encaps_pred_wS_data.txt",unpack=True)
    prediction_wSi, truth_wSi = np.genfromtxt("good_data/encaps_pred_wSi_data.txt",unpack=True)
    prediction_encaps, truth_encaps = np.genfromtxt("good_data/encaps_encaps_pred_data.txt", unpack=True)

    print('finished with reading data of encapsulated_RF.py ... \n')
    ######Energy PLOTS######
        # first prediction
    r2_1 = func.plot_hist2d(predictions,truth,min_energy,max_energy,bin_edge)
    plt.savefig("plots/RF/final/RF.pdf")
    plt.close()

        ## mean
    r2_1 = func.plot_hist2d(prediction_mean,truth_mean,min_energy,max_energy,bin_edge)
    plt.savefig("plots/RF/final/RF_mean.pdf")
    plt.close()

        ## median
    r2_1 = func.plot_hist2d(prediction_median,truth_median,min_energy,max_energy,bin_edge)
    plt.savefig("plots/RF/final/RF_median.pdf")
    plt.close()

        ## Intensity
    r2_1 = func.plot_hist2d(prediction_wI,truth_wI,min_energy,max_energy,bin_edge)
    plt.savefig("plots/RF/final/RF_wI.pdf")
    plt.close()

        ## sensitivity
    r2_1 = func.plot_hist2d(prediction_wS,truth_wS,min_energy,max_energy,bin_edge)
    plt.savefig("plots/RF/final/RF_wS.pdf")
    plt.close()

        ## size
    r2_1 = func.plot_hist2d(prediction_wSi,truth_wSi,min_energy,max_energy,bin_edge)
    plt.savefig("plots/RF/final/RF_wSi.pdf")
    plt.close()

        ## nested
    r2_1 = func.plot_hist2d(prediction_encaps,truth_encaps,min_energy,max_energy,bin_edge)
    plt.savefig("plots/RF/final/RF_encaps.pdf")
    plt.close()
    print("energy plots finished ... \n")


    ############### bias and std plot ################
        #Vergleich mean und median
    df = pd.DataFrame({'prediction':predictions, 'truth':truth})
    df_mean = pd.DataFrame({'prediction':prediction_mean,'truth':truth_mean})
    df_median = pd.DataFrame({'prediction':prediction_median,'truth':truth_median})
    bin_edge = np.logspace(np.log10(0.01),np.log10(max_energy),20)
    ax_bias = plt.gca()
    func.bias(df=df,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_bias=ax_bias,label='Schätzung',color="r")
    func.bias(df=df_mean,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_bias=ax_bias,label='Mittelwert',color="b")
    func.bias(df=df_median,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_bias=ax_bias,label='Median',color="g")
    ax_bias.set_xscale('log')
    ax_bias.set_ylabel("Verzerrung")
    ax_bias.set_xlabel(r'$E_{true}\, / \, TeV$')
    ax_bias.axhline(y=0,xmin=0,xmax=1,ls='--',c='grey')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/RF/final/RF_mean_bias.pdf")
    plt.close()


    ax_resolution = plt.gca()
    func.resolution(df=df,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_res=ax_resolution,label='Schätzung',color="r")
    func.resolution(df=df_mean,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_res=ax_resolution,label='Mittelwert',color="b")
    func.resolution(df=df_median,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_res=ax_resolution,label='Median',color="g")
    ax_resolution.set_xscale('log')
    ax_resolution.set_ylabel("IQA")
    ax_resolution.set_xlabel(r'$E_{true}\, / \, TeV$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/RF/final/RF_mean_resolution.pdf")
    plt.close()

        # Verleich der Gewichte
    #df = pd.DataFrame({'prediction':predictions, 'truth':truth})
    df_wI = pd.DataFrame({'prediction':prediction_wI,'truth':truth_wI})
    df_wS = pd.DataFrame({'prediction':prediction_wS,'truth':truth_wS})
    df_mean = pd.DataFrame({'prediction':prediction_mean,'truth':truth_mean})
    df_median = pd.DataFrame({'prediction':prediction_median,'truth':truth_median})
    #df_wSi = pd.DataFrame({'prediction':prediction_wSi,'truth':truth_wSi})
    bin_edge = np.logspace(np.log10(0.01),np.log10(max_energy),20)
    ax_bias = plt.gca()
    func.bias(df=df_mean,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_bias=ax_bias,label='Mittelwert',color="b")
    func.bias(df=df_wI,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_bias=ax_bias,label='Intensität',color="r")
    func.bias(df=df_wS,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_bias=ax_bias,label='Sensitivität',color="k")
    func.bias(df=df_median,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_bias=ax_bias,label='Median',color="g")
    ax_bias.set_xscale('log')
    ax_bias.set_ylabel("Verzerrung")
    ax_bias.set_xlabel(r'$E_{true}\, / \, TeV$')
    ax_bias.axhline(y=0,xmin=0,xmax=1,ls='--',c='grey')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/RF/final/RF_weights_bias.pdf")
    plt.close()


    ax_resolution = plt.gca()
    func.resolution(df=df_mean,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_res=ax_resolution,label='Mittelwert',color="b")
    func.resolution(df=df_wI,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_res=ax_resolution,label='Intensität',color="r")
    func.resolution(df=df_wS,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_res=ax_resolution,label='Sensitivität',color="k")
    func.resolution(df=df_median,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_res=ax_resolution,label='Median',color="g")
    ax_resolution.set_xscale('log')
    ax_resolution.set_ylabel("IQA")
    ax_resolution.set_xlabel(r'$E_{true}\, / \, TeV$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/RF/final/RF_weights_resolution.pdf")
    plt.close()

        # nested modell
    df_nested =  pd.DataFrame({'prediction':prediction_encaps,'truth':truth_encaps})
    bin_edge = np.logspace(np.log10(0.01),np.log10(max_energy),20)
    ax_bias = plt.gca()
    func.bias(df=df,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_bias=ax_bias,label='Schätzung',color="r")
    func.bias(df=df_mean,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_bias=ax_bias,label='Mittelwert',color="b")
    func.bias(df=df_nested,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_bias=ax_bias,label='verschachtelt',color="g")
    ax_bias.set_xscale('log')
    ax_bias.set_ylabel("Verzerrung")
    ax_bias.set_xlabel(r'$E_{true}\, / \, TeV$')
    ax_bias.axhline(y=0,xmin=0,xmax=1,ls='--',c='grey')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/RF/final/RF_nested_bias.pdf")
    plt.close()


    ax_resolution = plt.gca()
    func.resolution(df=df,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_res=ax_resolution,label='Schätzung',color="r")
    func.resolution(df=df_mean,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_res=ax_resolution,label='Mittelwert',color="b")
    func.resolution(df=df_nested,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_res=ax_resolution,label='verschachtelt',color="g")
    ax_resolution.set_xscale('log')
    ax_resolution.set_ylabel("IQA")
    ax_resolution.set_xlabel(r'$E_{true}\, / \, TeV$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/RF/final/RF_nested_resolution.pdf")
    plt.close()

    print("all plots of encapsulated_RF.py finished \n")


    ############################### Plots für trafo.py #####################################

    predictions, truth = np.genfromtxt("good_data/trafo_pred_data.txt",unpack=True)
    prediction_mean, truth_mean = np.genfromtxt("good_data/trafo_pred_mean_data.txt",unpack=True)
    prediction_encaps_trafo, truth_encaps_trafo = np.genfromtxt("good_data/trafo_encaps_pred_data.txt",unpack=True)

    bin_edge = np.logspace(np.log10(min_energy),np.log10(max_energy),300)
        #Energy Plots
    # first prediction
    r2_1 = func.plot_hist2d(predictions,truth,min_energy,max_energy,bin_edge)
    plt.savefig("plots/RF/final/trafo.pdf")
    plt.close()

    ## mean
    r2_1 = func.plot_hist2d(prediction_mean,truth_mean,min_energy,max_energy,bin_edge)
    plt.savefig("plots/RF/final/trafo_mean.pdf")
    plt.close()
    ## nested
    r2_1 = func.plot_hist2d(prediction_encaps_trafo,truth_encaps_trafo,min_energy,max_energy,bin_edge)
    plt.savefig("plots/RF/final/trafo_encaps.pdf")
    plt.close()
    print("energy plots finished ... \n")

        # Bias and resolution plots
    df_nested =  pd.DataFrame({'prediction':prediction_encaps,'truth':truth_encaps})
    df_nested_trafo =  pd.DataFrame({'prediction':prediction_encaps_trafo,'truth':truth_encaps_trafo})
    bin_edge = np.logspace(np.log10(0.01),np.log10(max_energy),20)
    ax_bias = plt.gca()

    func.bias(df=df,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_bias=ax_bias,label='Schätzung',color="r")
    func.bias(df=df_nested_trafo,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_bias=ax_bias,label='mit Transformation',color="b")
    func.bias(df=df_nested,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_bias=ax_bias,label='ohne Transformation',color="g")
    ax_bias.set_xscale('log')
    ax_bias.set_ylabel("Verzerrung")
    ax_bias.set_xlabel(r'$E_{true}\, / \, TeV$')
    ax_bias.axhline(y=0,xmin=0,xmax=1,ls='--',c='grey')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/RF/final/trafo_nested_bias.pdf")
    plt.close()


    ax_resolution = plt.gca()
    func.resolution(df=df,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_res=ax_resolution,label='Schätzung',color="r")
    func.resolution(df=df_nested_trafo,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_res=ax_resolution,label='mit Transformation',color="b")
    func.resolution(df=df_nested,bins=bin_edge,prediction_key='prediction',true_energy_key='truth',ax_res=ax_resolution,label='ohne Transformation',color="g")
    ax_resolution.set_xscale('log')
    ax_resolution.set_ylabel("IQA")
    ax_resolution.set_xlabel(r'$E_{true}\, / \, TeV$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/RF/final/trafo_nested_resolution.pdf")
    plt.close()

    print("all plots of encapsulated_RF.py finished \n")


if __name__ == '__main__' :
    plot()
