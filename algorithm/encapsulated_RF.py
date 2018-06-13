import itertools
import functions as func
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py as h5
import scipy as sc
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate, train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=argparse._textwrap.dedent('''\
                                    What do you want to calculate?
                                    ------------------------------
                                        decide with --steps:
                                        1: Just the prediction from the RFRegressor
                                        with mean scaled values.
                                        2: The mean weighted with intensity and distance to core.
                                        3: The encapsulated Tree with the mean as new
                                        attribute.
                                        - deafult: 3

                                        Decide between mean scaled values and not with --msv
                                            -default: True

                                        Decide if you want to consider the diffused gammas with --diffuse
                                            -default: True

                                        Decide how big your data should be with --size
                                            -you get --size events from gammas and diffuse gammas
                                             if you set --diffuse on True
                                            -default is 10000000
                                    '''))
parser.add_argument('--step', type=int, default=3)
parser.add_argument('--sv', type=bool, default=True, help="Wanna have the Scaled Value?")
parser.add_argument('--diffuse', type=bool, default=False, help="Wanna have the diffuse gammas?")
parser.add_argument('--size', type=int, default=10000000, help="How much data you want to enquire?")


def encaps_RF():
    args = parser.parse_args()

    data_size = args.size-1


    if(args.step > 0 & args.step < 4):

        data = func.reading_data(args.diffuse,data_size)



    print("Finished with reading Data ... \n")
    if(args.sv):

        #calculate the mean scaled
        data = func.calc_scaled_width_and_length(data)


        print("Finished with calculating Mean Scaled Values ... \n")

    if(args.step > 0 & args.step < 4):
        data = shuffle(data)
        #drop unimportant DATA
        data, droped_data = func.drop_data(data)
        truth = droped_data['mc_energy']

        #fit and predict
        RFr = RandomForestRegressor(max_depth=10, n_jobs=3,n_estimators=200)
        print("We use these attributes for the first RF: \n ",list(data))
        X=data.values
        y=truth.values
        predictions = cross_val_predict(RFr, X, y, cv=10)
        z=np.array([predictions,y])

        np.savetxt("data/encaps_pred_data.txt",z.T)

        print('RandomForestRegressor:\n\t Coefficient for determination: %.2f \n' % r2_score(predictions,truth.values),
                '\texplained_variance score: %.2f \n' % explained_variance_score(predictions,truth.values),
                '\tmean squared error: %.2f \n' % mean_squared_error(predictions,truth.values),
                "Finished with the first prediction ... \n")

    if(args.step > 1):
        # weighted mean over sensitivity
        telescope_sens = droped_data['telescope_type_name'].copy(deep=True)
        telescope_sens = telescope_sens.to_frame()
        truth_wS = droped_data['mc_energy']
        telescope_sens = pd.concat([telescope_sens,truth_wS],axis=1)
        mask = (telescope_sens['telescope_type_name'] == 'LST') & (telescope_sens['mc_energy'] > 3.0) # not in requiered energy range
        telescope_sens[mask] = 0.1
        mask = (telescope_sens['telescope_type_name'] == 'LST') & (telescope_sens['mc_energy']>0.15) & (telescope_sens['mc_energy']<3) #not in full sensitivity
        telescope_sens[mask] = 1
        mask = (telescope_sens['telescope_type_name'] == 'LST') & (telescope_sens['mc_energy']<0.15) # not in requiered energy range
        telescope_sens[mask] = 2
        mask = (telescope_sens['telescope_type_name'] == 'MST') & (telescope_sens['mc_energy'] > 50.0) # not in requiered energy range
        telescope_sens[mask] = 0.1
        mask = (telescope_sens['telescope_type_name'] == 'MST') & (telescope_sens['mc_energy'] < 0.08) # not in requiered energy range
        telescope_sens[mask] = 0.1
        mask = (telescope_sens['telescope_type_name'] == 'MST') & (telescope_sens['mc_energy']>5.0) & (telescope_sens['mc_energy']<50.0) #not in full sensitivity
        telescope_sens[mask] = 1
        mask = (telescope_sens['telescope_type_name'] == 'MST') & (telescope_sens['mc_energy']>0.08) & (telescope_sens['mc_energy']<0.15) #not in full sensitivity
        telescope_sens[mask] = 1
        mask = (telescope_sens['telescope_type_name'] == 'MST') & (telescope_sens['mc_energy']<5) & (telescope_sens['mc_energy']>0.15) # not in requiered energy range
        telescope_sens[mask] = 2
        mask = (telescope_sens['telescope_type_name'] == 'SST') & (telescope_sens['mc_energy'] > 300.0)
        telescope_sens[mask] = 0.1
        mask = (telescope_sens['telescope_type_name'] == 'SST') & (telescope_sens['mc_energy'] < 1.0)
        telescope_sens[mask] = 0.1
        mask = (telescope_sens['telescope_type_name'] == 'SST') & (telescope_sens['mc_energy']>1.0) & (telescope_sens['mc_energy']<5.0) #not in full sensitivity
        telescope_sens[mask] = 1
        mask = (telescope_sens['telescope_type_name'] == 'SST') & (telescope_sens['mc_energy']<300.0) & (telescope_sens['mc_energy']>5.0) # not in requiered energy range
        telescope_sens[mask] = 2
        telescope_sens = telescope_sens.drop('mc_energy',axis=1)
        pred = pd.DataFrame({'predicted_energy':predictions, 'mc_energy':
                            truth, 'weight': telescope_sens['telescope_type_name']})
        pred['weighted_data'] = pred['predicted_energy']*pred['weight']
        x = pred.groupby(level=list(['run_id','array_event_id']))
        prediction_wS = x['weighted_data'].sum()/x['weight'].sum()
        prediction_wS = prediction_wS.to_frame('predicted_energy')
        truth_wS = x['mc_energy'].mean()

        z=np.array([prediction_wS['predicted_energy'].values,truth_wS.values])
        np.savetxt("data/encaps_pred_wS_data.txt",z.T)




        print('RF with weighted mean(sensitivity):\n\t Coefficient for determination: %.2f \n' % r2_score(prediction_wS.values,truth_wS.values),
        '\texplained_variance score: %.2f \n' % explained_variance_score(prediction_wS.values,truth_wS.values),
        '\tmean squared error: %.2f \n' % mean_squared_error(prediction_wS.values,truth_wS.values))



    if(args.step == 3):
        # use the prediction_w_mean for another RF
        encaps_info = list(['num_triggered_telescopes','num_triggered_lst','num_triggered_mst','num_triggered_sst','total_intensity'])
        data_encaps = pd.concat([prediction_wS,data[encaps_info].drop_duplicates()],axis=1,join='inner')

        ######## neue Attribute berechnen #########

            ######### Mittelwert der Energien nur für die LST's ###########

        pred1 = pd.DataFrame({'predicted_energy':predictions, 'mc_energy':truth})
        pred1 = pred1.drop('mc_energy',axis=1)
        pred_lst = pred1[droped_data['telescope_type_name']=='LST']
        prediction_lst_max = pred_lst.groupby(level=list(['run_id','array_event_id'])).max()
        prediction_lst_min = pred_lst.groupby(level=list(['run_id','array_event_id'])).min()
        prediction_lst = pred_lst.groupby(level=list(['run_id','array_event_id'])).mean()
        prediction_lst_std =  pred_lst.groupby(level=list(['run_id','array_event_id'])).std()
        prediction_lst_max.columns = ['max_lst_pred']
        prediction_lst_min.columns = ['min_lst_pred']
        prediction_lst.columns = ['mean_lst_pred']
        prediction_lst_std.columns = ['std_lst_pred']

        pred_mst = pred1[droped_data['telescope_type_name']=='MST']
        prediction_mst_max = pred_mst.groupby(level=list(['run_id','array_event_id'])).max()
        prediction_mst_min = pred_mst.groupby(level=list(['run_id','array_event_id'])).min()
        prediction_mst = pred_mst.groupby(level=list(['run_id','array_event_id'])).mean()
        prediction_mst_std =  pred_mst.groupby(level=list(['run_id','array_event_id'])).std()
        prediction_mst.columns = ['mean_mst_pred']
        prediction_mst_std.columns = ['std_mst_pred']
        prediction_mst_max.columns = ['max_mst_pred']
        prediction_mst_min.columns = ['min_mst_pred']

        pred_sst = pred1[droped_data['telescope_type_name']=='SST']
        prediction_sst_max = pred_sst.groupby(level=list(['run_id','array_event_id'])).max()
        prediction_sst_min = pred_sst.groupby(level=list(['run_id','array_event_id'])).min()
        prediction_sst = pred_sst.groupby(level=list(['run_id','array_event_id'])).mean()
        prediction_sst_std =  pred_sst.groupby(level=list(['run_id','array_event_id'])).std()
        prediction_sst.columns = ['mean_sst_pred']
        prediction_sst_std.columns = ['std_sst_pred']
        prediction_sst_max.columns = ['max_sst_pred']
        prediction_sst_min.columns = ['min_sst_pred']

        data_encaps = pd.concat([data_encaps,prediction_lst,prediction_lst_std,prediction_mst,prediction_mst_std,
                                prediction_sst,prediction_sst_std,prediction_lst_min,prediction_lst_max,
                                prediction_mst_min,prediction_mst_max,prediction_sst_min,prediction_sst_max],axis=1)

        msl = data['scaled_length'].groupby(level=list(['run_id','array_event_id'])).mean()
        msw = data['scaled_width'].groupby(level=list(['run_id','array_event_id'])).mean()
        msl=msl.rename('mean_scaled_length')
        msw=msw.rename('mean_scaled_width')
        sl_std = data['scaled_length'].groupby(level=list(['run_id','array_event_id'])).std()
        sw_std = data['scaled_width'].groupby(level=list(['run_id','array_event_id'])).std()
        sl_std=sl_std.rename('std_scaled_length')
        sw_std=sw_std.rename('std_scaled_width')
        data_encaps = pd.concat([data_encaps,msl,msw,sl_std,sw_std],axis=1)

            ##### if there is no lst or sst or mst who has seen this event, I set the mean and std on 0
        data_encaps = data_encaps.fillna(0)
        data_encaps = pd.concat([data_encaps,truth_wS],axis=1)


        data_encaps = shuffle(data_encaps)
        truth_encaps = data_encaps['mc_energy'].copy(deep=True)
        data_encaps = data_encaps.drop('mc_energy', axis=1)

        #fit and pred
        RFr2 = RandomForestRegressor(max_depth=10, n_jobs=3,n_estimators=30)
        print("We use these attributes for the second RF: \n ",list(data_encaps))
        X=data_encaps.values
        y=truth_encaps.values
        predictions_encaps = cross_val_predict(RFr2, X, y, cv=10)

        z=np.array([predictions_encaps,truth_encaps.values])

        np.savetxt("data/encaps_encaps_pred_data.txt",z.T)

        print('encapsulated RF:\n\t Coefficient of determination: %.2f\n' % r2_score(predictions_encaps,truth_encaps.values),
        '\texplained_variance score: %.2f \n' % explained_variance_score(predictions_encaps,truth_encaps.values),
        '\tmean squared error: %.2f \n' % mean_squared_error(predictions_encaps,truth_encaps.values),
        "Finished with the encapsulated prediction \n")



        ############ tree interpretation of the second forest ###########################


        RF = RandomForestRegressor(max_depth=10,n_jobs=3,n_estimators=30)
        trainX,testX,trainY,testY = train_test_split(X,y)
        RF.fit(trainX,trainY)
        feature = RF.feature_importances_
        std = np.std([tree.feature_importances_ for tree in RF.estimators_],
                     axis=0)
        indices = np.argsort(feature)[::-1]
        names = list(data_encaps)
        # Print the feature ranking
        print("Feature ranking:")

        for f in range(X.shape[1]):
            print("%d. feature %s (%f)" % (f + 1, names[indices[f]], feature[indices[f]]))

        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), feature[indices],
               color="b", yerr=std[indices], align="center",)
        plt.xticks(range(X.shape[1]),[names[i] for i in indices],rotation=90)
        plt.xlim([-1, X.shape[1]])
        plt.tight_layout()
        plt.savefig("plots/feautureimportance_secondForest.jpg")
        plt.close()

        data=np.array([tree.feature_importances_ for tree in RF.estimators_])
        data=data[:,indices]
        position_ticks = np.arange(0,X.shape[1])+1
        plt.boxplot(data,notch=False)
        plt.xticks(position_ticks,[names[i] for i in indices],rotation=90)
        plt.tight_layout()
        plt.savefig("plots/feautureimportance_boxplot_secondForest.pdf")
        plt.close()






if __name__ == '__main__':
	encaps_RF()
