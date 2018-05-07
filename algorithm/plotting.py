import matplotlib.pyplot as plt
import functions as func
import numpy as np
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=argparse._textwrap.dedent('''\
                                    What do you want to plot?
                                    ------------------------------
                                        decide with --steps:
                                        1: plot the data from encapsulated_RF.py
                                        2: plot the data from RandomForestRegressor.py.
                                        0: plot all.
                                    ''')
parser.add_argument("--steps", type=int, default=0)

def plot():
    args = parser.parse_args()

    min_energy = 0.003
    max_energy = 340

    if(args.steps == 1 | args.steps == 0):
        ############PLotting of encapsulated_RF.py################


        #reading data

        predictions, truth = np.genfromtxt("data/encaps_pred_data.txt", unpack=True)
        prediction_w_mean, truth_w_mean = np.genfromtxt("data/encaps_pred_w_mean_data.txt", unpack=True)
        predictions_encaps, truth_encaps = np.genfromtxt("data/encaps_encaps_pred_data.txt", unpack=True)


        ######Energy PLOTS######
            #Plots without mean
        plt.subplot(211)
        r2_1 = func.plot_hist2d(predictions,truth,min_energy,max_energy)
        plt.title("RF(with MSV)(R2score: %.2f)" % r2_1)

            #weighted mean (intensity)
        plt.subplot(223)
        r2_2 = func.plot_hist2d(prediction_w_mean,truth_w_mean,min_energy,max_energy)
        plt.title("RFr w mean(inty)(%0.2f)" % r2_2 )

            #plots for encapsulated RF
        plt.subplot(224)
        r2_3 = func.plot_hist2d(predictions_encaps,truth_encaps,min_energy,max_energy)
        plt.title("encap RFr(%.2f)" % r2_3)
        plt.subplots_adjust(wspace=0.45,hspace=0.45)
        #plt.show()
        plt.savefig('plots/RF/mean_scaled/RF_Regression_MSV_all.pdf')
        plt.close()


        #####Error Plots#####
            #encapsulated RF
        func.plot_error(predictions_encaps,truth_encaps)
        plt.title('the error of the encapsulated RF for Energy estimation')
        #plt.show()
        plt.savefig('plots/RF/mean_scaled/RF_Regression_MSV_encaps_errors.pdf')
        plt.close()

            # without mean
        func.plot_error(predictions,truth)
        plt.title('error of RF(with MSV) for Energy estimation')
        plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV_error.pdf")
        plt.close()

            #with weighted mean (Intensity)
        func.plot_error(prediction_w_mean,truth_w_mean)
        plt.title('the error of the RF for Energy estimation with weighted mean(intensity)')
        #plt.show()
        plt.savefig('plots/RF/mean_scaled/RF_Regression_MSV_w_mean_error.pdf')
        plt.close()

        ######### relative Error #########
            #encapsulated RF
        func.plot_rel_error(predictions_encaps,truth_encaps)
        plt.title("The relative Error of the encapsulated RF regressor")
        plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV_encaps_relerror.pdf")
        plt.close()

            #without mean
        func.plot_rel_error(predictions,truth)
        plt.title("the ralative Error of the RF with MSV")
        plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV_relerror.pdf")
        plt.close()

            #with weighted mean(Intensity)
        func.plot_rel_error(prediction_w_mean,truth_w_mean)
        plt.title("The relative Error of the RF regression with MSV and weighted mean over Intensity")
        plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV_w_mean_relerror.pdf")
        plt.close()

        ######### true / pred ######

        func.plot_trueDIVpred(predictions,truth)
        plt.title("The Truth/Prediction of the RF with MSV")
        plt.savefig("plots/RF/mean_scaled/RF_Regression_MSV_truedivpred.pdf")
        plt.close()

        



    if(args.steps == 2 | args.steps == 0):

        ################ plotting of RandomForestRegressor.py ################

        # reading data

        predictions, truth = np.genfromtxt("data/RFr_pred_data.txt", unpack=True)
        prediction_mean, truth_mean = np.genfromtxt("data/RFr_pred_mean_data.txt", unpack=True)
        predictions_w_mean, truth_w_mean = np.genfromtxt("data/Rfr_pred_wI_mean_data.txt", unpack=True)
        predictions_w2_mean, truth_w2_mean = np.genfromtxt("data/RFr_pred_wT_mean_data.txt", unpack=True)


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
        plt.savefig('plots/RF/weighted/RF_Regression_all.pdf')
        plt.close()


        ######### Error Plots########
            #weighted mean (Telescope size)
        func.plot_error(prediction_w2_mean, truth_w2_mean)
        plt.title('the error of RF with weighted mean (telescope size)')
        #plt.show()
        plt.savefig('plots/RF/weighted/RF_Regression_errors_w2_mean.pdf')
        plt.close()

            #without mean
        func.plot_error(predictions,truth)
        plt.title('error of RF for Energy estimation')
        plt.savefig("plots/RF/weighted/RF_Regression_error.pdf")
        plt.close()

            #with mean
        func.plot_error(prediction_mean,truth_mean)
        plt.title('error of RF with mean for Energy estimation')
        plt.savefig("plots/RF/weighted/RF_Regression_mean_error.pdf")
        plt.close()

            #weighted mean (Intensity)
        func.plot_error(prediction_w_mean,truth_w_mean)
        plt.title('the error of the RF with weighted mean(intensity)')
        #plt.show()
        plt.savefig('plots/RF/weighted/RF_Regression_errors_w_mean.pdf')
        plt.close()

if __name__ == __main__ :
    plot()
