import itertools
import functions as func
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py as h5
import scipy as sc
import dask.dataframe as dd
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

                                        Decide between mean scaled values and not with --sv
                                            -default: True

                                        Decide if you want to consider the diffused gammas with --diffuse
                                            -default: True

                                        Decide how big your data should be with --size
                                            -you get --size events from gammas and diffuse gammas
                                             if you set --diffuse on True
                                            -default is all data
                                    '''))
parser.add_argument('--step', type=int, default=3)
parser.add_argument('--sv', type=bool, default=False, help="Wanna have the Scaled Value?")
parser.add_argument('--diffuse', type=bool, default=False, help="Wanna have the diffuse gammas?")
parser.add_argument('--size', type=int, default=-1, help="How much data you want to enquire?")


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

        #data['weight'] = droped_data['telescope_type_name']
        truth = droped_data['mc_energy']


            #### plotte Truth gegen num_triggered_sst #######
        #x1 = truth.values
        #y1 = data['num_triggered_sst'].values
        #plt.plot(x1,y1,'.')
        #plt.ylabel("num_triggered_sst")
        #plt.xlabel("mc_energy in TeV")
        #plt.xscale('log')
        #plt.savefig("plots/sst_mc.jpg")
        #plt.close()


        #fit and predict
        RFr = RandomForestRegressor(max_depth=10, n_jobs=-1,n_estimators=100, oob_score=True)
        X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(data.index,truth.index,test_size=0.66)
        X_train = data.loc[X_train_i]
        X_test = data.loc[X_test_i]
        y_train = truth.loc[y_train_i]
        y_test = truth.loc[y_test_i]
        #X_train = X_train.drop('weight',axis=1)
        #weight = X_test['weight']
        #X_test = X_test.drop('weight',axis=1)

        X1 = X_train.values
        X2 = X_test.values
        y1 = y_train.values
        print("We use these attributes for the first RF: \n ",list(X_train))
        RFr.fit(X1, y1)
                ############### overfitting ####################
        print("The oob_score is: ",RFr.oob_score_)

                ############# feature importance ################
        feature = RFr.feature_importances_
        std = np.std([tree.feature_importances_ for tree in RFr.estimators_],
                     axis=0)
        indices = np.argsort(feature)[::-1]
        names = list(data)
        # Print the feature ranking
        print("Feature ranking:")

        for f in range(X1.shape[1]):
            print("%d. feature %s (%f)" % (f + 1, names[indices[f]], feature[indices[f]]))


        data1=np.array([tree.feature_importances_ for tree in RFr.estimators_])
        data1=data1[:,indices]
        position_ticks = np.arange(0,X1.shape[1])+1
        plt.boxplot(data1,notch=False)
        plt.xticks(position_ticks,[names[i] for i in indices],rotation=90)
        plt.tight_layout()
        plt.savefig("plots/feautureimportance_boxplot_firstForest.pdf")
        plt.close()

            ################# prediction###############
        predictions = RFr.predict(X2)
        print("Trainiert mit:",y_train.shape[0]," \t Getestet mit: ",y_test.shape[0])

        z=np.array([predictions,y_test.values])
        np.savetxt("data/encaps_pred_data.txt",z.T)

        print('RandomForestRegressor:\n\t Coefficient for determination: %.2f \n' % r2_score(predictions,y_test.values),
                '\texplained_variance score: %.2f \n' % explained_variance_score(predictions,y_test.values),
                '\tmean squared error: %.2f \n' % mean_squared_error(predictions,y_test.values),
                "Finished with the first prediction ... \n")

    if(args.step > 1):
        # weighted mean over sensitivity
        ###telescope_sens = weight
        ###telescope_sens = telescope_sens.to_frame()
        ###telescope_sens['pred'] = predictions
        ###telescope_sens2 = telescope_sens.copy(deep=True)
        ###mask = (telescope_sens['weight'] == 'LST') & (telescope_sens['pred'] > 3.0) # not in requiered energy range
        ###telescope_sens2[mask] = 0.1
        ###mask = (telescope_sens['weight'] == 'LST') & (telescope_sens['pred']>0.15) & (telescope_sens['pred']<3) #not in full sensitivity
        ###telescope_sens2[mask] = 1
        ###mask = (telescope_sens['weight'] == 'LST') & (telescope_sens['pred']<0.15) # not in requiered energy range
        ###telescope_sens2[mask] = 2
        ###mask = (telescope_sens['weight'] == 'MST') & (telescope_sens['pred'] > 50.0) # not in requiered energy range
        ###telescope_sens2[mask] = 0.1
        ###mask = (telescope_sens['weight'] == 'MST') & (telescope_sens['pred'] < 0.08) # not in requiered energy range
        ###telescope_sens2[mask] = 0.1
        ###mask = (telescope_sens['weight'] == 'MST') & (telescope_sens['pred']>5.0) & (telescope_sens['pred']<50.0) #not in full sensitivity
        ###telescope_sens2[mask] = 1
        ###mask = (telescope_sens['weight'] == 'MST') & (telescope_sens['pred']>0.08) & (telescope_sens['pred']<0.15) #not in full sensitivity
        ###telescope_sens2[mask] = 1
        ###mask = (telescope_sens['weight'] == 'MST') & (telescope_sens['pred']<5) & (telescope_sens['pred']>0.15) # not in requiered energy range
        ###telescope_sens2[mask] = 2
        ###mask = (telescope_sens['weight'] == 'SST') & (telescope_sens['pred'] > 300.0)
        ###telescope_sens2[mask] = 0.1
        ###mask = (telescope_sens['weight'] == 'SST') & (telescope_sens['pred'] < 1.0)
        ###telescope_sens2[mask] = 0.1
        ###mask = (telescope_sens['weight'] == 'SST') & (telescope_sens['pred']>1.0) & (telescope_sens['pred']<5.0) #not in full sensitivity
        ###telescope_sens2[mask] = 1
        ###mask = (telescope_sens['weight'] == 'SST') & (telescope_sens['pred']<300.0) & (telescope_sens['pred']>5.0) # not in requiered energy range
        ###telescope_sens2[mask] = 2
        ###telescope_sens2 = telescope_sens2.drop('pred',axis=1)
        ###pred = pd.DataFrame({'predicted_energy':predictions,'mc_energy':y_test})
        ###pred['weight']=telescope_sens2['weight']
        ###pred['weighted_data'] = pred['predicted_energy']*pred['weight']
        ###x = pred.groupby(level=['run_id','array_event_id'],sort=False)
        ###prediction_wS = x['weighted_data'].sum()/x['weight'].sum()
        ###prediction_wS = prediction_wS.to_frame('predicted_energy')
        ###y_test1 = y_test.reset_index()
        ###truth_wS = y_test1.drop_duplicates()
        ###truth_wS = truth_wS.set_index(['run_id','array_event_id'])
        data_w = X_test.copy(deep=True)
        data_w['mc_energy'] = y_test
        data_w['predictions'] = predictions
        data_w = func.weighted_mean_over_ID(data_w['intensity'],data_w)
        data_w = data_w.reset_index()
        prediction_wI = data_w[['weighted_prediction','array_event_id','run_id','mc_energy']]
        prediction_wI = prediction_wI.drop_duplicates()

        #std_wS = std_wS.reset_index()
        #plt.plot(std_wS.index,std_wS['predicted_energy'].values,'.')
        #plt.ylabel("sigma")
        #plt.xlabel('event')
        #plt.savefig("plots/std.jpg")
        #plt.close()



        z=np.array([prediction_wI['weighted_prediction'].values,prediction_wI['mc_energy'].values])
        np.savetxt("data/encaps_pred_wS_data.txt",z.T)

        print('RF with weighted mean(intensity):\n\t Coefficient for determination: %.2f \n' % r2_score(prediction_wI['weighted_prediction'].values,prediction_wI['mc_energy'].values),
        '\texplained_variance score: %.2f \n' % explained_variance_score(prediction_wI['weighted_prediction'].values,prediction_wI['mc_energy'].values),
        '\tmean squared error: %.2f \n' % mean_squared_error(prediction_wI['weighted_prediction'].values,prediction_wI['mc_energy'].values))



    if(args.step == 3):
        # use the prediction_w_mean for another RF
        encaps_info = list(['num_triggered_telescopes','num_triggered_lst','num_triggered_mst','num_triggered_sst','total_intensity'])
        data_encaps = X_test[encaps_info].reset_index().drop_duplicates()
        print(prediction_wI.shape)
        data_encaps = pd.merge(data_encaps,prediction_wI, on=['run_id','array_event_id'])
        print(data_encaps.shape)

        ######## neue Attribute berechnen #########

            ######### Mittelwert der Energien nur für die LST's ###########
        pred = data_w[['predictions','array_event_id','run_id','telescope_type_id']]
        telescope_type = pred['telescope_type_id'].copy(deep=True)
        pred = pred.drop('telescope_type_id',axis=1)
        telescope_type[telescope_type==1] = 'LST'
        telescope_type[telescope_type==2] = 'MST'
        telescope_type[telescope_type==3] = 'SST'
        pred_lst = pred[telescope_type=='LST']
        prediction_lst_max = pred_lst.groupby(by=list(['run_id','array_event_id'])).max().reset_index()
        prediction_lst_min = pred_lst.groupby(by=list(['run_id','array_event_id'])).min().reset_index()
        prediction_lst = pred_lst.groupby(by=list(['run_id','array_event_id'])).mean().reset_index()
        prediction_lst_std =  pred_lst.groupby(by=list(['run_id','array_event_id'])).std().reset_index()
        prediction_lst_max = prediction_lst_max.rename(columns = {'predictions':'max_lst_pred'})
        prediction_lst_min = prediction_lst_max.rename(columns = {'predictions':'min_lst_pred'})
        prediction_lst = prediction_lst.rename(columns = {'prediction':'mean_lst_pred'})
        prediction_lst_std = prediction_lst_std.rename(columns = {'predictions':'std_lst_pred'})

        pred_mst = pred[telescope_type=='MST']
        prediction_mst_max = pred_mst.groupby(by=list(['run_id','array_event_id'])).max().reset_index()
        prediction_mst_min = pred_mst.groupby(by=list(['run_id','array_event_id'])).min().reset_index()
        prediction_mst = pred_mst.groupby(by=list(['run_id','array_event_id'])).mean().reset_index()
        prediction_mst_std =  pred_mst.groupby(by=list(['run_id','array_event_id'])).std().reset_index()
        prediction_mst = prediction_mst.rename(columns={'predictions':'mean_mst_pred'})
        prediction_mst_std = prediction_mst_std.rename(columns={'predictions':'std_mst_pred'})
        prediction_mst_max = prediction_mst_max.rename(columns = {'predictions':'max_mst_pred'})
        prediction_mst_min = prediction_mst_min.rename(columns = {'predictions':'min_mst_pred'})

        pred_sst = pred[telescope_type=='SST']
        prediction_sst_max = pred_sst.groupby(by=list(['run_id','array_event_id'])).max().reset_index()
        prediction_sst_min = pred_sst.groupby(by=list(['run_id','array_event_id'])).min().reset_index()
        prediction_sst = pred_sst.groupby(by=list(['run_id','array_event_id'])).mean().reset_index()
        prediction_sst_std =  pred_sst.groupby(by=list(['run_id','array_event_id'])).std().reset_index()
        prediction_sst = prediction_sst.rename(columns = {'predictions':'mean_sst_pred'})
        prediction_sst_std = prediction_sst_std.rename(columns = {'predictions':'std_sst_pred'})
        prediction_sst_max = prediction_sst_max.rename(columns = {'predictions':'max_sst_pred'})
        prediction_sst_min = prediction_sst_min.rename(columns = {'predictions':'min_sst_pred'})
        data_encaps =data_encaps.merge(prediction_lst,on=['array_event_id','run_id']).merge(prediction_lst_std,on=['array_event_id','run_id']).merge(prediction_mst,on=['array_event_id','run_id']).merge(prediction_mst_std,
            on=['array_event_id','run_id']).merge(prediction_sst,on=['array_event_id','run_id']).merge(prediction_sst_std,
            on=['array_event_id','run_id']).merge(prediction_lst_min,on=['array_event_id','run_id']).merge(prediction_lst_max,
            on=['array_event_id','run_id']).merge(prediction_mst_min,on=['array_event_id','run_id']).merge(prediction_mst_max,
            on=['array_event_id','run_id']).merge(prediction_sst_min,on=['array_event_id','run_id']).merge(prediction_sst_max,
            on=['array_event_id','run_id'])

        if(args.sv):
            msl = X_test['scaled_length'].groupby(by=list(['run_id','array_event_id'])).mean().reset_index()
            msw = X_test['scaled_width'].groupby(by=list(['run_id','array_event_id'])).mean().reset_index()
            msl=msl.rename('mean_scaled_length')
            msw=msw.rename('mean_scaled_width')
            sl_std = X_test['scaled_length'].groupby(by=list(['run_id','array_event_id'])).std().reset_index()
            sw_std = X_test['scaled_width'].groupby(by=list(['run_id','array_event_id'])).std().reset_index()
            sl_std=sl_std.rename('std_scaled_length')
            sw_std=sw_std.rename('std_scaled_width')
            data_encaps = data_encaps.merge(msl,on=['array_event_id','run_id']).merge(msw,on=['array_event_id','run_id']).merge(sl_std,
                on=['array_event_id','run_id']).merge(sw_std,on=['array_event_id','run_id'])



            ##### if there is no lst or sst or mst who has seen this event, I set the mean and std on 0
        data_encaps = data_encaps.fillna(0)


        data_encaps = shuffle(data_encaps)
        truth_encaps = data_encaps['mc_energy'].copy(deep=True)
        data_encaps = data_encaps.drop(['mc_energy','array_event_id','run_id'], axis=1)
        #fit and pred
        RFr2 = RandomForestRegressor(max_depth=10, n_jobs=-1,n_estimators=100,oob_score=True)
        print("We use these attributes for the second RF: \n ",list(data_encaps))
        X_train, X_test, y_train, y_test = train_test_split(data_encaps,truth_encaps,test_size=0.5)
        RFr2.fit(X_train,y_train)

                ############### overfitting ####################
        print("The oob_score is: ",RFr2.oob_score_)

                ############# feature importance ################
        feature = RFr2.feature_importances_
        std = np.std([tree.feature_importances_ for tree in RFr2.estimators_],
                     axis=0)
        indices = np.argsort(feature)[::-1]
        names = list(data_encaps)
        # Print the feature ranking
        print("Feature ranking:")

        for f in range(X_train.shape[1]):
            print("%d. feature %s (%f)" % (f + 1, names[indices[f]], feature[indices[f]]))


        data2=np.array([tree.feature_importances_ for tree in RFr2.estimators_])
        data2=data2[:,indices]
        position_ticks = np.arange(0,X_train.shape[1])+1
        plt.boxplot(data2,notch=False)
        plt.xticks(position_ticks,[names[i] for i in indices],rotation=90)
        plt.tight_layout()
        plt.savefig("plots/feautureimportance_boxplot_secondForest.pdf")
        plt.close()


                ####### Predictions ################
        prediction_encaps = RFr2.predict(X_test)

        print("Trainiert mit:",y_train.shape[0]," \t Getestet mit: ",y_test.shape[0])

        z=np.array([prediction_encaps,y_test.values])
        np.savetxt("data/encaps_encaps_pred_data.txt",z.T)

        print('encapsulated RF:\n\t Coefficient of determination: %.2f\n' % r2_score(prediction_encaps,y_test.values),
        '\texplained_variance score: %.2f \n' % explained_variance_score(prediction_encaps,y_test.values),
        '\tmean squared error: %.2f \n' % mean_squared_error(prediction_encaps,y_test.values),
        "Finished with the encapsulated prediction \n")






if __name__ == '__main__':
	encaps_RF()
