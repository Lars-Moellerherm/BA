import matplotlib.pyplot as plt
import functions as func
import numpy as np
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=argparse._textwrap.dedent('''\
                                    What do you want to plot?
                                    ------------------------------
                                        decide with --steps:
                                        1: plot the data of encapsulated_RF.py.
                                        2: plot the data of RandomForestRegressor.py.
                                        0: plot all.
                                    '''))
parser.add_argument('--steps', type=int, default=0)


def plot():
    args = parser.parse_args()

    min_energy = 0.003
    max_energy = 340
    bin_edge = np.logspace(np.log10(min_energy),np.log10(max_energy),35)

      ############PLotting of encapsulated_RF.py################
    if(args.steps == 1) or (args.steps == 0):
        #reading data
        predictions, truth = np.genfromtxt("good_data/encaps_pred_data.txt", unpack=True)
        prediction_w_mean, truth_w_mean = np.genfromtxt("good_data/encaps_pred_w_mean_data.txt", unpack=True)
        prediction_w2_mean, truth_w2_mean = np.genfromtxt("good_data/encaps_pred_w2_mean_data.txt",unpack=True)
        predictions_encaps, truth_encaps = np.genfromtxt("good_data/encaps_encaps_pred_data.txt", unpack=True)

        print('finished with reading data of encapsulated_RF.py ... \n')
        ######Energy PLOTS######
            #Plots without mean
        plt.subplot(211)
        r2_1 = func.plot_hist2d(predictions,truth,min_energy,max_energy,bin_edge)
        plt.title("RF(with MSV)(R2score: %.2f)" % r2_1)

            #weighted mean (intensity)
        plt.subplot(223)
        r2_2 = func.plot_hist2d(prediction_w_mean,truth_w_mean,min_energy,max_energy,bin_edge)
        plt.title("RFr w mean(inty)(%0.2f)" % r2_2 )

            #plots for encapsulated RF
        plt.subplot(224)
        r2_3 = func.plot_hist2d(predictions_encaps,truth_encaps,min_energy,max_energy,bin_edge)
        plt.title("encap RFr(%.2f)" % r2_3)
        plt.subplots_adjust(wspace=0.45,hspace=0.45)
        #plt.show()
        plt.savefig('plots/RF/mean_scaled/RF_Regression_MSV_all.jpg')
        plt.close()

            #plot for weighted mean(dist to core)
        r2_4 = func.plot_hist2d(prediction_w2_mean,truth_w2_mean,min_energy,max_energy,bin_edge)
        plt.title("RFr w mean(dist_to_core)(R2score: %.2f)" % r2_4)
        plt.savefig("plots/RF/mean_scaled/Rf_Regression_MSV_w2_mean.jpg")
        plt.close()

        print("energy plots finished ... \n")

        #####Error Plots#####
            #encapsulated RF
        func.plot_error(predictions_encaps,truth_encaps)
        plt.title('the error of the encapsulated RF for Energy estimation')
        #plt.show()
        plt.savefig('plots/RF/mean_scaled/RF_Regression_MSV_encaps_errors.jpg')
        plt.close()

            # without mean
        print(np.size(predictions_encaps))
        func.plot_error(predictions,truth)
        plt.title('error of RF(with MSV) for Energy estimation')
        plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV_error.jpg")
        plt.close()

            #with weighted mean (Intensity)
        func.plot_error(prediction_w_mean,truth_w_mean)
        plt.title('the error of the RF with w mean(intensity)')
        #plt.show()
        plt.savefig('plots/RF/mean_scaled/RF_Regression_MSV_w_mean_error.jpg')
        plt.close()

            #with weighted mean (distance to core)
        func.plot_error(prediction_w2_mean,truth_w2_mean)
        plt.title('the error of the RF with w mean(dist to core)')
        #plt.show()
        plt.savefig('plots/RF/mean_scaled/RF_Regression_MSV_w2_mean_error.jpg')
        plt.close()

        print("error plots finished ... \n")

        ######### relative Error #########
            #encapsulated RF
        mu , sigma = func.plot_rel_error(predictions_encaps,truth_encaps)
        plt.title(r"rel Error of the encapsulated RF regressor ($\mu = %.2f$ ; $\sigma_{\mu} = %.2f$)" %  (mu,sigma))
        plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV_encaps_relerror.jpg")
        plt.close()

            #without mean
        mu , sigma = func.plot_rel_error(predictions,truth)
        plt.title("rel Error of the RF with MSV ($\mu = %.2f$ ; $\sigma_{\mu} = %.2f$)" %  (mu,sigma))
        plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV_relerror.jpg")
        plt.close()

            #with weighted mean(Intensity)
        mu , sigma = func.plot_rel_error(prediction_w_mean,truth_w_mean)
        plt.title("rel Error of the RF with MSV and w mean (Intensity) ($\mu = %.2f$ ; $\sigma_{\mu} = %.2f$)" %  (mu,sigma))
        plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV_w_mean_relerror.jpg")
        plt.close()

            #weighted mean( dist to core)
        mu, sigma = func.plot_rel_error(prediction_w2_mean,truth_w2_mean)
        plt.title("rel Error of the RF with MSV and w mean(dist to core) ($\mu = %.2f$ ; $\sigma_{\mu} = %.2f$)" %  (mu,sigma))
        plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV_w2_mean_relerror.jpg")
        plt.close()

        print('relative error plots finished ... \n')
        ######### true / pred ######
            #without mean
        mu, sigma = func.plot_trueDIVpred(predictions,truth)
        plt.title("The Truth/Prediction of the RF with MSV ($\mu = %.2f$ ; $\sigma_{\mu} = %.2f$)" %  (mu,sigma))
        plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV_truedivpred.jpg")
        plt.close()

            #with weighted mean(intensity)
        mu, sigma = func.plot_trueDIVpred(prediction_w_mean,truth_w_mean)
        plt.title("The Truth/Prediction of the RF with MSV and w mean(intensity) ($\mu = %.2f$ ; $\sigma_{\mu} = %.2f$)" %  (mu,sigma))
        plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV_w_mean_truedivpred.jpg")
        plt.close()

            #with weighted mean(dist to core)
        mu, sigma = func.plot_trueDIVpred(prediction_w2_mean,truth_w2_mean)
        plt.title("The Truth/Prediction of the RF with MSV and w mean(dist to core) ($\mu = %.2f$ ; $\sigma_{\mu} = %.2f$)" %  (mu,sigma))
        plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV_w2_mean_truedivpred.jpg")
        plt.close()

            #encapsulated RF
        mu, sigma = func.plot_trueDIVpred(predictions_encaps,truth_encaps)
        plt.title("The Truth/Prediction of the encaps RF with MSV ($\mu = %.2f$ ; $\sigma_{\mu} = %.2f$)" %  (mu,sigma))
        plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV_encaps_truedivpred.jpg")
        plt.close()

        print("true/pred plots finished ... \n")

        ############Std of each bin######################
            #without mean
        func.plot_std_der_bins(predictions,bin_edge)
        plt.title("Std of each bin for RF r with MSV")
        plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV_std.jpg")
        plt.close()

            #with weighted mean(intensity)
        func.plot_std_der_bins(prediction_w_mean,bin_edge)
        plt.title("Std of each bin for RF r with MSV and w mean(Intensity)")
        plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV_w_mean_std.jpg")
        plt.close()

            #weighted mean(dist to core)
        func.plot_std_der_bins(prediction_w2_mean,bin_edge)
        plt.title("Std of each bin for RF r with MSV and w mean(dist to core)")
        plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV_w2_mean_std.jpg")
        plt.close()

            #encapsulated RF
        func.plot_std_der_bins(predictions_encaps,bin_edge)
        plt.title("Std of each bin for encapsulated RF r ")
        plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV_encaps_std.jpg")
        plt.close()

        print("std of bins plots finished ... \n")

        print("all plots of encapsulated_RF.py finished \n")


    ################ plotting of RandomForestRegressor.py ################
    if(args.steps == 2) or (args.steps == 0):

        # reading data

        predictions, truth = np.genfromtxt("good_data/RFr_pred_data.txt", unpack=True)
        prediction_mean, truth_mean = np.genfromtxt("good_data/RFr_pred_mean_data.txt", unpack=True)
        predictions_w_mean, truth_w_mean = np.genfromtxt("good_data/Rfr_pred_wI_mean_data.txt", unpack=True)
        predictions_w2_mean, truth_w2_mean = np.genfromtxt("good_data/RFr_pred_wT_mean_data.txt", unpack=True)


        #######Energy PLOTS#######
            #Plots without mean
        plt.subplot(221)
        r2_1 = func.plot_hist2d(predictions,truth,min_energy,max_energy)
        plt.title("RFr(R2score: %.2f)" % r2_1)

            #Plots with mean
        plt.subplot(222)
        r2_2 = func.plot_hist2d(prediction_mean,truth_mean,min_energy,max_energy)
        plt.title("RFr mean(%.2f)" % r2_2)

                #intensity
        plt.subplot(223)
        r2_3 = func.plot_hist2d(prediction_w_mean,truth_w_mean,min_energy,max_energy)
        plt.title("RFr w mean(inty)(%0.2f)" % r2_3 )

                #telescope size
        plt.subplot(224)
        r2_4 = func.plot_hist2d(prediction_w2_mean,truth_w2_mean,min_energy,max_energy)
        plt.title("RFr w mean(telsize) (%.2f)" % r2_4)
        plt.subplots_adjust(wspace=0.45,hspace=0.45)
        #plt.show()
        plt.savefig('plots/RF/weighted/RF_Regression_all.jpg')
        plt.close()


        ######### Error Plots########
            #weighted mean (Telescope size)
        func.plot_error(prediction_w2_mean, truth_w2_mean)
        plt.title('the error of RF with weighted mean (telescope size)')
        #plt.show()
        plt.savefig('plots/RF/weighted/RF_Regression_errors_w2_mean.jpg')
        plt.close()

            #without mean
        func.plot_error(predictions,truth)
        plt.title('error of RF for Energy estimation')
        plt.savefig("plots/RF/weighted/RF_Regression_error.jpg")
        plt.close()

            #with mean
        func.plot_error(prediction_mean,truth_mean)
        plt.title('error of RF with mean for Energy estimation')
        plt.savefig("plots/RF/weighted/RF_Regression_mean_error.jpg")
        plt.close()

            #weighted mean (Intensity)
        func.plot_error(prediction_w_mean,truth_w_mean)
        plt.title('the error of the RF with weighted mean(intensity)')
        #plt.show()
        plt.savefig('plots/RF/weighted/RF_Regression_errors_w_mean.jpg')
        plt.close()

if __name__ == '__main__' :
    plot()
