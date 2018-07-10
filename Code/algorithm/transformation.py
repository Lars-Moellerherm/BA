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

                                        Decide between mean scaled values and not with --sv
                                            -default: True

                                        Decide if you want to consider the diffused gammas with --diffuse
                                            -default: True

                                        Decide how big your data should be with --size
                                            -you get --size events from gammas and diffuse gammas
                                             if you set --diffuse on True
                                            -default is all data
                                    '''))
parser.add_argument('--sv', type=bool, default=True, help="Wanna have the Scaled Value?")
parser.add_argument('--diffuse', type=bool, default=True, help="Wanna have the diffuse gammas?")
parser.add_argument('--size', type=int, default=-1, help="How much data you want to enquire?")


def RF_trafo():
    args = parser.parse_args()

    data_size = args.size-1

    data = func.reading_data(args.diffuse,data_size)

    print("Finished with reading Data ... \n")

    if(args.sv):

        #calculate the mean scaled
        data = func.calc_scaled_width_and_length(data)
        data = data.drop(['scaled_width','scaled_length'],axis=1)


        print("Finished with calculating Mean Scaled Values ... \n")



    data = shuffle(data)
    #drop unimportant DATA
    data, droped_data = func.drop_data(data)
    #data['weight'] = droped_data['telescope_type_name']
    truth = data[['mc_energy','array_event_id','run_id']]
    data = data.drop('mc_energy',axis=1)
    #fit and predict
    RFr = RandomForestRegressor(max_depth=10, n_jobs=-1,n_estimators=100, oob_score=True, max_features='sqrt')
    train_i, test_i = train_test_split(data[['array_event_id','run_id']],test_size=0.66)
    X_train = data.loc[data[['array_event_id','run_id']].isin(train_i)[data[['array_event_id','run_id']].isin(train_i)==True].dropna().index]
    X_test = data.loc[data[['array_event_id','run_id']].isin(test_i)[data[['array_event_id','run_id']].isin(test_i)==True].dropna().index]
    y_train = truth.loc[truth[['array_event_id','run_id']].isin(train_i)[truth[['array_event_id','run_id']].isin(train_i)==True].dropna().index]
    y_test = truth.loc[truth[['array_event_id','run_id']].isin(test_i)[truth[['array_event_id','run_id']].isin(test_i)==True].dropna().index]

    if(args.sv):
        #calculate the mean scaled
        X_train = func.calc_scaled_width_and_length(X_train)
        X_test = func.calc_scaled_width_and_length(X_test)

    X1 = X_train.drop(['array_event_id','run_id'],axis=1).values
    X2 = X_test.drop(['array_event_id','run_id'],axis=1).values
    y1 = y_train.drop(['array_event_id','run_id'],axis=1).values
    y2 = y_test.drop(['array_event_id','run_id'],axis=1).values
    print("We use these attributes for the first RF: \n ",list(X_train.drop(['array_event_id','run_id'],axis=1)))

    ##################### Trafo #############################
    y1=np.log(y1+3)
    RFr.fit(X1, y1)

            ############### overfitting ####################
    print("The oob_score is: ",RFr.oob_score_)



            ############# feature importance ################
    feature = RFr.feature_importances_
    indices = np.argsort(feature)[::-1]
    names = list(X_train.drop(['array_event_id','run_id'],axis=1))
    names = func.translate(names,1)
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X1.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, names[indices[f]], feature[indices[f]]))
    data1=np.array([tree.feature_importances_ for tree in RFr.estimators_])
    data1=data1[:,indices]
    position_ticks = np.arange(0,X1.shape[1])+1
    plt.boxplot(data1,notch=False)
    plt.xticks(position_ticks,[names[i] for i in indices],rotation=90)
    plt.ylabel('Wichtigkeit')
    plt.tight_layout()
    plt.savefig("plots/feautureimportance_boxplot_trafo_firstForest.pdf")
    plt.close()


        ################# prediction###############
    predictions = RFr.predict(X2)
    print("Trainiert mit:",y_train.shape[0]," \t Getestet mit: ",y_test.shape[0])
    pred = pd.DataFrame({'prediction':predictions, 'mc_energy':y_test['mc_energy'], 'array_event_id':y_test['array_event_id'], 'run_id':y_test['run_id']})

    ######## Rücktransformation ###########
    predictions_rück = np.exp(predictions)-3

    z=np.array([predictions_rück,y2[:,0]])
    np.savetxt("data/trafo_pred_data.txt",z.T)


    print('RandomForestRegressor:\n\t Coefficient for determination: %.2f \n' % r2_score(predictions_rück,y2[:,0]),
            '\texplained_variance score: %.2f \n' % explained_variance_score(predictions_rück,y2[:,0]),
            '\tmean squared error: %.2f \n' % mean_squared_error(predictions_rück,y2[:,0]),
            "Finished with the first prediction ... \n")


    ######### gewichteter und nicht gewichteter Mittelwert ######################
    X_test_w = X_test.set_index(['run_id','array_event_id'])
    pred_w = pred.set_index(['run_id','array_event_id'])
    data_w = pd.concat([X_test_w,pred_w],axis=1).reset_index()
    truth_grouped = y_test.drop_duplicates().set_index(['run_id','array_event_id'])
            ################ Mean and Median ##################
    x_grouped = data_w[['prediction','array_event_id','run_id']].groupby(by=['run_id','array_event_id'])
    pred_mean = x_grouped.mean()
    pred_mean.columns = ['mean_prediction']
    pred_mean = pd.concat([pred_mean,truth_grouped], axis=1)

    ####Rücktrafo ######
    pred_mean_rück = np.exp(pred_mean['mean_prediction'])-3
    pred_mean_rück = pd.concat([pred_mean_rück,truth_grouped], axis=1)
    z=np.array([pred_mean_rück['mean_prediction'].values,pred_mean_rück['mc_energy'].values])
    np.savetxt("data/trafo_pred_mean_data.txt",z.T)


    print('RF with mean:\n\t Coefficient for determination: %.2f \n' % r2_score(pred_mean_rück['mean_prediction'].values,pred_mean_rück['mc_energy'].values),
    '\texplained_variance score: %.2f \n' % explained_variance_score(pred_mean_rück['mean_prediction'].values,pred_mean_rück['mc_energy'].values),
    '\tmean squared error: %.2f \n' % mean_squared_error(pred_mean_rück['mean_prediction'].values,pred_mean_rück['mc_energy'].values))



    # use the prediction_median for another RF
    encaps_info = ['width','length','num_triggered_telescopes','num_triggered_lst','num_triggered_mst','num_triggered_sst','total_intensity','array_event_id','run_id']
    data_encaps = X_test[encaps_info]
    pred_mean = pred_mean.reset_index()
    data_encaps = data_encaps.merge(pred_mean,on=['run_id','array_event_id'])
    train_i2, test_i2 = train_test_split(data_encaps[['array_event_id','run_id']],test_size=0.5)
    X2_train = data_encaps.loc[data_encaps[['array_event_id','run_id']].isin(train_i2)[data_encaps[['array_event_id','run_id']].isin(train_i2)==True].dropna().index]
    X2_test = data_encaps.loc[data_encaps[['array_event_id','run_id']].isin(test_i2)[data_encaps[['array_event_id','run_id']].isin(test_i2)==True].dropna().index]
    if(args.sv):
        X2_train = func.calc_mean_scaled_width_and_length(X2_train)
        X2_test = func.calc_mean_scaled_width_and_length(X2_test)
    ######## neue Attribute berechnen #########
        ######### Mittelwert der Energien nur für die LST's ###########
    pred = data_w[['prediction','array_event_id','run_id','telescope_type_id']]
    telescope_type = pred['telescope_type_id'].copy(deep=True)
    pred = pred.drop('telescope_type_id',axis=1)
    pred_lst = pred[telescope_type==1]
    prediction_lst_max = pred_lst.groupby(by=list(['run_id','array_event_id'])).max().reset_index()
    prediction_lst_min = pred_lst.groupby(by=list(['run_id','array_event_id'])).min().reset_index()
    prediction_lst = pred_lst.groupby(by=list(['run_id','array_event_id'])).mean().reset_index()
    prediction_lst_std =  pred_lst.groupby(by=list(['run_id','array_event_id'])).std().reset_index()
    prediction_lst_max = prediction_lst_max.rename(columns = {'prediction':'max_lst_pred'})
    prediction_lst_min = prediction_lst_min.rename(columns = {'prediction':'min_lst_pred'})
    prediction_lst = prediction_lst.rename(columns = {'prediction':'mean_lst_pred'})
    prediction_lst_std = prediction_lst_std.rename(columns = {'prediction':'std_lst_pred'})
    pred_mst = pred[telescope_type==2]
    prediction_mst_max = pred_mst.groupby(by=list(['run_id','array_event_id'])).max().reset_index()
    prediction_mst_min = pred_mst.groupby(by=list(['run_id','array_event_id'])).min().reset_index()
    prediction_mst = pred_mst.groupby(by=list(['run_id','array_event_id'])).mean().reset_index()
    prediction_mst_std =  pred_mst.groupby(by=list(['run_id','array_event_id'])).std().reset_index()
    prediction_mst = prediction_mst.rename(columns={'prediction':'mean_mst_pred'})
    prediction_mst_std = prediction_mst_std.rename(columns={'prediction':'std_mst_pred'})
    prediction_mst_max = prediction_mst_max.rename(columns = {'prediction':'max_mst_pred'})
    prediction_mst_min = prediction_mst_min.rename(columns = {'prediction':'min_mst_pred'})
    pred_sst = pred[telescope_type==3]
    prediction_sst_max = pred_sst.groupby(by=list(['run_id','array_event_id'])).max().reset_index()
    prediction_sst_min = pred_sst.groupby(by=list(['run_id','array_event_id'])).min().reset_index()
    prediction_sst = pred_sst.groupby(by=list(['run_id','array_event_id'])).mean().reset_index()
    prediction_sst_std =  pred_sst.groupby(by=list(['run_id','array_event_id'])).std().reset_index()
    prediction_sst = prediction_sst.rename(columns = {'prediction':'mean_sst_pred'})
    prediction_sst_std = prediction_sst_std.rename(columns = {'prediction':'std_sst_pred'})
    prediction_sst_max = prediction_sst_max.rename(columns = {'prediction':'max_sst_pred'})
    prediction_sst_min = prediction_sst_min.rename(columns = {'prediction':'min_sst_pred'})
    X2_train = X2_train.reset_index()
    X2_test = X2_test.reset_index()
    X2_train = X2_train.merge(prediction_lst,how='left',on=['run_id','array_event_id']).merge(prediction_lst_std,
                    how='left',on=['run_id','array_event_id']).merge(prediction_mst,how='left',on=['run_id','array_event_id']).merge(prediction_mst_std,
                    how='left',on=['run_id','array_event_id']).merge(prediction_sst,how='left',on=['run_id','array_event_id']).merge(prediction_sst_std,
                    how='left',on=['run_id','array_event_id']).merge(prediction_lst_min,how='left',on=['run_id','array_event_id']).merge(prediction_lst_max,
                    how='left',on=['run_id','array_event_id']).merge(prediction_mst_min,how='left',on=['run_id','array_event_id']).merge(prediction_mst_max,
                    how='left',on=['run_id','array_event_id']).merge(prediction_sst_min,how='left',on=['run_id','array_event_id']).merge(prediction_sst_max,
                    how='left',on=['run_id','array_event_id'])
    X2_test = X2_test.merge(prediction_lst,how='left',on=['run_id','array_event_id']).merge(prediction_lst_std,
                    how='left',on=['run_id','array_event_id']).merge(prediction_mst,how='left',on=['run_id','array_event_id']).merge(prediction_mst_std,
                    how='left',on=['run_id','array_event_id']).merge(prediction_sst,how='left',on=['run_id','array_event_id']).merge(prediction_sst_std,
                    how='left',on=['run_id','array_event_id']).merge(prediction_lst_min,how='left',on=['run_id','array_event_id']).merge(prediction_lst_max,
                    how='left',on=['run_id','array_event_id']).merge(prediction_mst_min,how='left',on=['run_id','array_event_id']).merge(prediction_mst_max,
                    how='left',on=['run_id','array_event_id']).merge(prediction_sst_min,how='left',on=['run_id','array_event_id']).merge(prediction_sst_max,
                    how='left',on=['run_id','array_event_id'])
        ##### if there is no lst or sst or mst who has seen this event, I set the mean and std on 0 and the std is Nan if there is just one prediction
    X2_test = X2_test.fillna(0)
    X2_test = shuffle(X2_test)
    y2_test = X2_test[['mc_energy']].copy(deep=True)
    X2_test = X2_test.drop(['mc_energy','array_event_id','run_id'], axis=1)
    X2_train = X2_train.fillna(0)
    X2_train = shuffle(X2_train)
    y2_train = X2_train[['mc_energy']].copy(deep=True)
    X2_train = X2_train.drop(['mc_energy','array_event_id','run_id'], axis=1)
    #fit and pred
    RFr2 = RandomForestRegressor(max_depth=10, n_jobs=-1,n_estimators=100,oob_score=True, max_features='sqrt')
    print("We use these attributes for the second RF: \n ",list(X2_train))
    ########### Trafo ############
    y2_train = np.log(y2_train+3)
    RFr2.fit(X2_train.values,y2_train.values)
            ############### overfitting ####################
    print("The oob_score is: ",RFr2.oob_score_)
            ############# feature importance ################
    feature = RFr2.feature_importances_
    std = np.std([tree.feature_importances_ for tree in RFr2.estimators_],
                 axis=0)
    indices = np.argsort(feature)[::-1]
    names = list(X2_train)
    names = func.translate(names,2)
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X2_train.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, names[indices[f]], feature[indices[f]]))
    data2=np.array([tree.feature_importances_ for tree in RFr2.estimators_])
    data2=data2[:,indices]
    position_ticks = np.arange(0,X2_train.shape[1])+1
    plt.boxplot(data2,notch=False)
    plt.xticks(position_ticks,[names[i] for i in indices],rotation=90)
    plt.ylabel('Wichtigkeit')
    plt.tight_layout()
    plt.savefig("plots/feautureimportance_boxplot_trafo_secondForest.pdf")
    plt.close()
            ####### Predictions ################
    prediction_encaps = RFr2.predict(X2_test.values)
    print("Trainiert mit:",y2_train.shape[0]," \t Getestet mit: ",y2_test.shape[0])
        ############ Rücktrafo ###########
    prediction_encaps = np.exp(prediction_encaps)-3

    print("Trainiert mit:",y2_train.shape[0]," \t Getestet mit: ",y2_test.shape[0])
    z=np.array([prediction_encaps,y2_test['mc_energy'].values])
    np.savetxt("data/trafo_encaps_pred_data.txt",z.T)

    print('encapsulated with median_prediction RF:\n\t Coefficient of determination: %.2f\n' % r2_score(prediction_encaps,y2_test.values),
        '\texplained_variance score: %.2f \n' % explained_variance_score(prediction_encaps,y2_test.values),
        '\tmean squared error: %.2f \n' % mean_squared_error(prediction_encaps,y2_test.values),
        "Finished with the encapsulated prediction \n")







if __name__ == '__main__':
	RF_trafo()
