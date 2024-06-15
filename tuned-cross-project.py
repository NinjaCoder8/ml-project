import math
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from mlxtend.regressor import StackingRegressor
from xgboost import XGBRegressor, XGBRFRegressor
import time
import concurrent.futures


def read_data(folders):
    mergers=[]
    for i in range(0,len(folders)):
        files=[]
        for filename in os.listdir("data/"+folders[i]):
            if '.csv' in filename:
                files.append(filename)
        merged=pd.read_csv('data/'+folders[i]+"/"+files[0])
        for j in range(1,len(files)):
            df=pd.read_csv('data/'+folders[i]+"/"+files[j])
            merged=merged._append(df)
        mergers.append(merged)
    merged=mergers[0]
    for i in range(1,len(mergers)):
        merged=merged._append(mergers[i])

    merged=merged.drop(['name','version','name.1'], axis='columns')

    test_features=merged.drop(['bug'], axis='columns')
    test_label=merged['bug']

    test_features=test_features.values.reshape(-1,20)
    test_label=test_label.values.reshape(-1,1)

    return test_features,test_label,folders[0]



def get_data(folders):
    mergers=[]
    for i in range(0,len(folders)):
        files=[]
        for filename in os.listdir("data/"+folders[i]):
            if '.csv' in filename:
                files.append(filename)
        merged=pd.read_csv('data/'+folders[i]+"/"+files[0])
        for j in range(1,len(files)):
            df=pd.read_csv('data/'+folders[i]+"/"+files[j])
            merged=merged._append(df)
        mergers.append(merged)
    merged=mergers[0]
    for i in range(1,len(mergers)):
        merged=merged._append(mergers[i])

    merged=merged.drop(['name','version','name.1'], axis='columns')

    train_features=merged.drop(['bug'], axis='columns')
    train_label=merged['bug']

    train_features=train_features.values.reshape(-1,20)
    train_label=train_label.values.reshape(-1,1)

    return train_features,train_label


def mean_relative_error(actual, predicted):
    rel_err=[]
    actual,predicted=np.array(actual), np.array(predicted)
    for i in range(0,len(actual)):
        err=abs(actual[i]-predicted[i])/(actual[i]+1)
        rel_err.append(err)
    return np.mean(rel_err)

def measure_completeness(actual, predicted):
    actual,predicted=np.array(actual), np.array(predicted)
    truth=np.sum(actual)
    preds=np.sum(predicted)
    return preds/truth

def predl(actual,predicted,l):
    chosen=0
    actual,predicted=np.array(actual), np.array(predicted)
    for i in range(0,len(actual)):
        a=(actual[i]-actual[i]*l,actual[i]+actual[i]*l)
        if predicted[i]>=a[0] and predicted[i]<=a[1]:
            chosen=chosen+1
    return chosen/len(actual)

def hyper(x_train, y_train,filename):
    parameters_svr=[]
    parameters_rfr=[]
    parameters_sv=[]
    parameters_dt=[]
    parameters_rf=[]
    parameters_kn=[]
    parameters_et=[]
    result_svr=[]
    result_rfr=[]
    result_sv=[]
    result_dt=[]
    result_rf=[]
    result_kn=[]
    result_et=[]
    for seed in range(0,20):
        print("Tuning Run: ",seed+1," ",filename," started.")
        params_tree = {
            'n_estimators':[50,126],
            'min_samples_leaf':[30,50],
            'max_depth':[20,50]
            }

        params_d_tree = {
            'max_depth':[20,50]
            }

        params_knn = {
            'n_neighbors':[5,10]
            }

        params_svr ={
            'C':[5,10],
            'gamma':[0.001,0.1,1]
            }

        params_xgb = {
            'max_depth':[10,50],
            'min_child_weight':[1,6]
            }

        params_2 = {
            'svr__C':[0.001,0.1,1,5,10],
            'svr__gamma':[0.001, 0.1, 0.5,1],
            'extratreesregressor__n_estimators':[50,100,126,200],
            'extratreesregressor__min_samples_leaf':[20,30,50,10],
            'kneighborsregressor__n_neighbors':[5,7,10,15],
            'randomforestregressor__max_depth':[10,20,30,50],
            'randomforestregressor__n_estimators':[50,100,126,300],
            'meta_regressor__C':[0.001,0.1,1,5,10],
            'meta_regressor__gamma':[0.001,0.1,0.5,1],
            'meta_regressor__kernel':['rbf','linear']
            }

        params_1 = {
            'svr__C':[0.001,0.1,1,5,10],
            'svr__gamma':[0.001, 0.1, 0.5,1],
            'extratreesregressor__n_estimators':[50,100,126,200],
            'extratreesregressor__min_samples_leaf':[20,30,50,10],
            'kneighborsregressor__n_neighbors':[5,7,10,15],
            'randomforestregressor__max_depth':[10,20,30,50],
            'randomforestregressor__n_estimators':[50,100,126,300],
            'meta_regressor__max_depth':[10,20,30,50],
            'meta_regressor__n_estimators':[20,50,100,150,500]
            }

        #print('DTR')
        rand_search_dtr = RandomizedSearchCV(DecisionTreeRegressor(), params_d_tree, cv=10,random_state=seed,n_jobs=-1)
        rand_search_dtr.fit(x_train, y_train)
        best_tuned_dt_m = rand_search_dtr.best_params_
        best_tuned_dt_s = rand_search_dtr.best_score_
        parameters_dt.append(best_tuned_dt_m)
        result_dt.append(best_tuned_dt_s)

        #print('RFR')
        rand_search_rfr = RandomizedSearchCV(RandomForestRegressor(), params_tree, cv=10,random_state=seed,n_jobs=-1)
        rand_search_rfr.fit(x_train, y_train)
        best_tuned_rf_m = rand_search_rfr.best_params_
        best_tuned_rf_s = rand_search_rfr.best_score_
        parameters_rf.append(best_tuned_rf_m)
        result_rf.append(best_tuned_rf_s)


        #print('ETR')
        rand_search_etr = RandomizedSearchCV(ExtraTreesRegressor(), params_tree, cv=10,random_state=seed,n_jobs=-1)
        rand_search_etr.fit(x_train, y_train)
        best_tuned_et_m = rand_search_etr.best_params_
        best_tuned_et_s = rand_search_etr.best_score_
        parameters_et.append(best_tuned_et_m)
        result_et.append(best_tuned_et_s)

        #print('KNN')
        rand_search_knn = RandomizedSearchCV(KNeighborsRegressor(), params_knn, cv=10,random_state=seed,n_jobs=-1)
        rand_search_knn.fit(x_train, y_train)
        best_tuned_kn_m = rand_search_knn.best_params_
        best_tuned_kn_s = rand_search_knn.best_score_
        parameters_kn.append(best_tuned_kn_m)
        result_kn.append(best_tuned_kn_s)

        #print('SVR')
        rand_search_svr = RandomizedSearchCV(SVR(), params_svr, cv=10,random_state=seed,n_jobs=-1)
        rand_search_svr.fit(x_train, y_train)
        best_tuned_sv_m = rand_search_svr.best_params_
        best_tuned_sv_s = rand_search_svr.best_score_
        parameters_sv.append(best_tuned_sv_m)
        result_sv.append(best_tuned_sv_s)

        #print('SR')
        reg1=SVR()
        reg2=ExtraTreesRegressor()
        reg3=KNeighborsRegressor()
        reg4=RandomForestRegressor()
        rand_search_sr = RandomizedSearchCV(StackingRegressor(regressors=[reg1,reg2,reg3,reg4], meta_regressor=SVR()), params_2, cv=10,random_state=seed, n_jobs=-1)
        rand_search_sr.fit(x_train, y_train)
        best_tuned_sr_m = rand_search_sr.best_params_
        best_tuned_sr_s = rand_search_sr.best_score_
        parameters_svr.append(best_tuned_sr_m)
        result_svr.append(best_tuned_sr_s)

        #print('SR2')
        reg1=SVR()
        reg2=ExtraTreesRegressor()
        reg3=KNeighborsRegressor()
        reg4=RandomForestRegressor()
        rand_search_sr2 = RandomizedSearchCV(StackingRegressor(regressors=[reg1,reg2,reg3,reg4], meta_regressor=RandomForestRegressor()), params_1, cv=10,random_state=seed, n_jobs=-1)
        rand_search_sr2.fit(x_train, y_train)
        best_tuned_sr2_m = rand_search_sr2.best_params_
        best_tuned_sr2_s = rand_search_sr2.best_score_
        parameters_rfr.append(best_tuned_sr2_m)
        result_rfr.append(best_tuned_sr2_s)
        print("Tuning Run: ",seed+1," ",filename," finished.")

    best_pics={
        'svr':parameters_sv[result_sv.index(max(result_sv))],
        'dtr':parameters_dt[result_dt.index(max(result_dt))],
        'etr':parameters_et[result_et.index(max(result_et))],
        'rfr':parameters_rf[result_rf.index(max(result_rf))],
        'knn':parameters_kn[result_kn.index(max(result_kn))],
        'svr_meta':parameters_svr[result_svr.index(max(result_svr))],
        'rfr_meta':parameters_rfr[result_rfr.index(max(result_rfr))]
    }
    return best_pics

def run_techniques(train_folders,test_folders):
    t0 = time.time()
    test_features,test_label,filename=read_data(test_folders)
    train_features, train_label=get_data(train_folders)


    import warnings
    warnings.filterwarnings("ignore")

    train_label=np.ravel(train_label,order='C')
    best=hyper(train_features, train_label,filename)

    stravgmre,stravgmretest=[],[]
    stravgmae,stravgmaetest=[],[]
    stravgrmse,stravgrmsetest=[],[]
    stravgmse,stravgmsetest=[],[]
    stravgcomp,stravgcomptest=[],[]
    stravgrsq,stravgrsqtest=[],[]
    stravgpredl,stravgpredltest=[],[]

    stravgmre2,stravgmretest2=[],[]
    stravgmae2,stravgmaetest2=[],[]
    stravgrmse2,stravgrmsetest2=[],[]
    stravgmse2,stravgmsetest2=[],[]
    stravgcomp2,stravgcomptest2=[],[]
    stravgrsq2,stravgrsqtest2=[],[]
    stravgpredl2,stravgpredltest2=[],[]

    dtravgmre,dtravgmretest=[],[]
    dtravgmae,dtravgmaetest=[],[]
    dtravgrmse,dtravgrmsetest=[],[]
    dtravgmse,dtravgmsetest=[],[]
    dtravgcomp,dtravgcomptest=[],[]
    dtravgrsq,dtravgrsqtest=[],[]
    dtravgpredl,dtravgpredltest=[],[]

    rfravgmre,rfravgmretest=[],[]
    rfravgmae,rfravgmaetest=[],[]
    rfravgrmse,rfravgrmsetest=[],[]
    rfravgmse,rfravgmsetest=[],[]
    rfravgcomp,rfravgcomptest=[],[]
    rfravgrsq,rfravgrsqtest=[],[]
    rfravgpredl,rfravgpredltest=[],[]

    etravgmre,etravgmretest=[],[]
    etravgmae,etravgmaetest=[],[]
    etravgrmse,etravgrmsetest=[],[]
    etravgmse,etravgmsetest=[],[]
    etravgcomp,etravgcomptest=[],[]
    etravgrsq,etravgrsqtest=[],[]
    etravgpredl,etravgpredltest=[],[]

    knnavgmre,knnavgmretest=[],[]
    knnavgmae,knnavgmaetest=[],[]
    knnavgrmse,knnavgrmsetest=[],[]
    knnavgmse,knnavgmsetest=[],[]
    knnavgcomp,knnavgcomptest=[],[]
    knnavgrsq,knnavgrsqtest=[],[]
    knnavgpredl,knnavgpredltest=[],[]

    svravgmre,svravgmretest=[],[]
    svravgmae,svravgmaetest=[],[]
    svravgrmse,svravgrmsetest=[],[]
    svravgmse,svravgmsetest=[],[]
    svravgcomp,svravgcomptest=[],[]
    svravgrsq,svravgrsqtest=[],[]
    svravgpredl,svravgpredltest=[],[]

    bavgmre,bavgmretest=[],[]
    bavgmae,bavgmaetest=[],[]
    bavgrmse,bavgrmsetest=[],[]

    for run in range(0,20):
        print("Testing Run: ",run+1," ",filename," started.")

        kf=KFold(n_splits=10,shuffle=True,random_state=run)
        i=1

        strmre,strmretest=[],[]
        strmae,strmaetest=[],[]
        strrmse,strrmsetest=[],[]
        strmse,strmsetest=[],[]
        strcomp,strcomptest=[],[]
        strrsq,strrsqtest=[],[]
        strpredl,strpredltest=[],[]

        strmre2,strmretest2=[],[]
        strmae2,strmaetest2=[],[]
        strrmse2,strrmsetest2=[],[]
        strmse2,strmsetest2=[],[]
        strcomp2,strcomptest2=[],[]
        strrsq2,strrsqtest2=[],[]
        strpredl2,strpredltest2=[],[]

        dtrmre,dtrmretest=[],[]
        dtrmae,dtrmaetest=[],[]
        dtrrmse,dtrrmsetest=[],[]
        dtrmse,dtrmsetest=[],[]
        dtrcomp,dtrcomptest=[],[]
        dtrrsq,dtrrsqtest=[],[]
        dtrpredl,dtrpredltest=[],[]

        rfrmre,rfrmretest=[],[]
        rfrmae,rfrmaetest=[],[]
        rfrrmse,rfrrmsetest=[],[]
        rfrmse,rfrmsetest=[],[]
        rfrcomp,rfrcomptest=[],[]
        rfrrsq,rfrrsqtest=[],[]
        rfrpredl,rfrpredltest=[],[]

        etrmre,etrmretest=[],[]
        etrmae,etrmaetest=[],[]
        etrrmse,etrrmsetest=[],[]
        etrmse,etrmsetest=[],[]
        etrcomp,etrcomptest=[],[]
        etrrsq,etrrsqtest=[],[]
        etrpredl,etrpredltest=[],[]

        knnmre,knnmretest=[],[]
        knnmae,knnmaetest=[],[]
        knnrmse,knnrmsetest=[],[]
        knnmse,knnmsetest=[],[]
        knncomp,knncomptest=[],[]
        knnrsq,knnrsqtest=[],[]
        knnpredl,knnpredltest=[],[]

        svrmre,svrmretest=[],[]
        svrmae,svrmaetest=[],[]
        svrrmse,svrrmsetest=[],[]
        svrmse,svrmsetest=[],[]
        svrcomp,svrcomptest=[],[]
        svrrsq,svrrsqtest=[],[]
        svrpredl,svrpredltest=[],[]

        bmre,bmretest=[],[]
        bmae,bmaetest=[],[]
        brmse,brmsetest=[],[]

        for train_index,test_index in kf.split(train_features):
            #print('fold number: ', i)

            x_train,x_test=train_features[train_index],train_features[test_index]
            y_train,y_test=train_label[train_index],train_label[test_index]


            #print('DTR')
            dtr = DecisionTreeRegressor(max_depth=best['dtr']['max_depth'])
            dtr.fit(x_train, y_train)
            y_pred = dtr.predict(test_features)
            test_pred = dtr.predict(x_test)
            #testing errors
            dtrmretest.append(mean_relative_error(y_test,test_pred))
            dtrmaetest.append(metrics.mean_absolute_error(y_test,test_pred))
            dtrrmsetest.append(math.sqrt(metrics.mean_squared_error(y_test,test_pred)))
            dtrmsetest.append(metrics.mean_squared_error(y_test,test_pred))
            dtrcomptest.append(measure_completeness(y_test,test_pred))
            dtrpredltest.append(predl(y_test,test_pred,0.3))
            dtrrsqtest.append(metrics.r2_score(y_test,test_pred))
            #validation errors
            dtrmre.append(mean_relative_error(test_label,y_pred))
            dtrmae.append(metrics.mean_absolute_error(test_label,y_pred))
            dtrrmse.append(math.sqrt(metrics.mean_squared_error(test_label,y_pred)))
            dtrmse.append(metrics.mean_squared_error(test_label,y_pred))
            dtrcomp.append(measure_completeness(test_label,y_pred))
            dtrpredl.append(predl(test_label,y_pred,0.3))
            dtrrsq.append(metrics.r2_score(test_label,y_pred))

            #print('RFR')
            rfr = RandomForestRegressor(n_estimators=best['rfr']['n_estimators'],min_samples_leaf=best['rfr']['min_samples_leaf'],max_depth=best['rfr']['max_depth'])
            rfr.fit(x_train, y_train)
            y_pred = rfr.predict(test_features)
            test_pred = rfr.predict(x_test)
            #testing errors
            rfrmretest.append(mean_relative_error(y_test,test_pred))
            rfrmaetest.append(metrics.mean_absolute_error(y_test,test_pred))
            rfrrmsetest.append(math.sqrt(metrics.mean_squared_error(y_test,test_pred)))
            rfrmsetest.append(metrics.mean_squared_error(y_test,test_pred))
            rfrcomptest.append(measure_completeness(y_test,test_pred))
            rfrpredltest.append(predl(y_test,test_pred,0.3))
            rfrrsqtest.append(metrics.r2_score(y_test,test_pred))
            #validation errors
            rfrmre.append(mean_relative_error(test_label,y_pred))
            rfrmae.append(metrics.mean_absolute_error(test_label,y_pred))
            rfrrmse.append(math.sqrt(metrics.mean_squared_error(test_label,y_pred)))
            rfrmse.append(metrics.mean_squared_error(test_label,y_pred))
            rfrcomp.append(measure_completeness(test_label,y_pred))
            rfrpredl.append(predl(test_label,y_pred,0.3))
            rfrrsq.append(metrics.r2_score(test_label,y_pred))

            #print('ETR')
            etr = ExtraTreesRegressor(n_estimators=best['etr']['n_estimators'], min_samples_leaf=best['etr']['min_samples_leaf'], max_depth=best['etr']['max_depth'])
            etr.fit(x_train, y_train)
            y_pred = etr.predict(test_features)
            test_pred = etr.predict(x_test)
            #testing errors
            etrmretest.append(mean_relative_error(y_test,test_pred))
            etrmaetest.append(metrics.mean_absolute_error(y_test,test_pred))
            etrrmsetest.append(math.sqrt(metrics.mean_squared_error(y_test,test_pred)))
            etrmsetest.append(metrics.mean_squared_error(y_test,test_pred))
            etrcomptest.append(measure_completeness(y_test,test_pred))
            etrpredltest.append(predl(y_test,test_pred,0.3))
            etrrsqtest.append(metrics.r2_score(y_test,test_pred))
            #validation errors
            etrmre.append(mean_relative_error(test_label,y_pred))
            etrmae.append(metrics.mean_absolute_error(test_label,y_pred))
            etrrmse.append(math.sqrt(metrics.mean_squared_error(test_label,y_pred)))
            etrmse.append(metrics.mean_squared_error(test_label,y_pred))
            etrcomp.append(measure_completeness(test_label,y_pred))
            etrpredl.append(predl(test_label,y_pred,0.3))
            etrrsq.append(metrics.r2_score(test_label,y_pred))

            #print('KNN')
            knn = KNeighborsRegressor(n_neighbors=best['knn']['n_neighbors'])
            knn.fit(x_train, y_train)
            y_pred = knn.predict(test_features)
            test_pred = knn.predict(x_test)
            #testing errors
            knnmretest.append(mean_relative_error(y_test,test_pred))
            knnmaetest.append(metrics.mean_absolute_error(y_test,test_pred))
            knnrmsetest.append(math.sqrt(metrics.mean_squared_error(y_test,test_pred)))
            knnmsetest.append(metrics.mean_squared_error(y_test,test_pred))
            knncomptest.append(measure_completeness(y_test,test_pred))
            knnpredltest.append(predl(y_test,test_pred,0.3))
            knnrsqtest.append(metrics.r2_score(y_test,test_pred))
            #validation errors
            knnmre.append(mean_relative_error(test_label,y_pred))
            knnmae.append(metrics.mean_absolute_error(test_label,y_pred))
            knnrmse.append(math.sqrt(metrics.mean_squared_error(test_label,y_pred)))
            knnmse.append(metrics.mean_squared_error(test_label,y_pred))
            knncomp.append(measure_completeness(test_label,y_pred))
            knnpredl.append(predl(test_label,y_pred,0.3))
            knnrsq.append(metrics.r2_score(test_label,y_pred))

            #print('SVR')
            svr = SVR(gamma=best['svr']['gamma'],C=best['svr']['C'])
            svr.fit(x_train, y_train)
            y_pred = svr.predict(test_features)
            test_pred = svr.predict(x_test)
            #testing errors
            svrmretest.append(mean_relative_error(y_test,test_pred))
            svrmaetest.append(metrics.mean_absolute_error(y_test,test_pred))
            svrrmsetest.append(math.sqrt(metrics.mean_squared_error(y_test,test_pred)))
            svrmsetest.append(metrics.mean_squared_error(y_test,test_pred))
            svrcomptest.append(measure_completeness(y_test,test_pred))
            svrpredltest.append(predl(y_test,test_pred,0.3))
            svrrsqtest.append(metrics.r2_score(y_test,test_pred))
            #validation errors
            svrmre.append(mean_relative_error(test_label,y_pred))
            svrmae.append(metrics.mean_absolute_error(test_label,y_pred))
            svrrmse.append(math.sqrt(metrics.mean_squared_error(test_label,y_pred)))
            svrmse.append(metrics.mean_squared_error(test_label,y_pred))
            svrcomp.append(measure_completeness(test_label,y_pred))
            svrpredl.append(predl(test_label,y_pred,0.3))
            svrrsq.append(metrics.r2_score(test_label,y_pred))

            t01 = time.time()
            #print('SR')
            reg1=SVR(gamma=best['svr_meta']['svr__gamma'],C=best['svr_meta']['svr__C'])
            reg2=ExtraTreesRegressor(n_estimators=best['svr_meta']['extratreesregressor__n_estimators'],min_samples_leaf=best['svr_meta']['extratreesregressor__min_samples_leaf'])
            reg3=KNeighborsRegressor(n_neighbors=best['svr_meta']['kneighborsregressor__n_neighbors'])
            reg4=RandomForestRegressor(n_estimators=best['svr_meta']['randomforestregressor__n_estimators'],max_depth=best['svr_meta']['randomforestregressor__max_depth'])
            rsr = StackingRegressor(regressors=[reg1,reg2,reg3,reg4], meta_regressor=SVR(kernel=best['svr_meta']['meta_regressor__kernel'],gamma=best['svr_meta']['meta_regressor__gamma'],C=best['svr_meta']['meta_regressor__C']))
            rsr.fit(x_train, y_train)
            y_pred = rsr.predict(test_features)
            test_pred = rsr.predict(x_test)
            #print('testing')
            #testing errors
            strmretest.append(mean_relative_error(y_test,test_pred))
            strmaetest.append(metrics.mean_absolute_error(y_test,test_pred))
            strrmsetest.append(math.sqrt(metrics.mean_squared_error(y_test,test_pred)))
            strmsetest.append(metrics.mean_squared_error(y_test,test_pred))
            strcomptest.append(measure_completeness(y_test,test_pred))
            strpredltest.append(predl(y_test,test_pred,0.3))
            strrsqtest.append(metrics.r2_score(y_test,test_pred))
            #print('validation')
            #validation errors
            strmre.append(mean_relative_error(test_label,y_pred))
            strmae.append(metrics.mean_absolute_error(test_label,y_pred))
            strrmse.append(math.sqrt(metrics.mean_squared_error(test_label,y_pred)))
            strmse.append(metrics.mean_squared_error(test_label,y_pred))
            strcomp.append(measure_completeness(test_label,y_pred))
            strpredl.append(predl(test_label,y_pred,0.3))
            strrsq.append(metrics.r2_score(test_label,y_pred))
            t11 = time.time()
            total1=t11-t01
            
            #print('SR2')
            t02 = time.time()
            reg1=SVR(gamma=best['rfr_meta']['svr__gamma'],C=best['rfr_meta']['svr__C'])
            reg2=ExtraTreesRegressor(n_estimators=best['rfr_meta']['extratreesregressor__n_estimators'],min_samples_leaf=best['rfr_meta']['extratreesregressor__min_samples_leaf'])
            reg3=KNeighborsRegressor(n_neighbors=best['rfr_meta']['kneighborsregressor__n_neighbors'])
            reg4=RandomForestRegressor(n_estimators=best['rfr_meta']['randomforestregressor__n_estimators'],max_depth=best['rfr_meta']['randomforestregressor__max_depth'])
            fsr = StackingRegressor(regressors=[reg1,reg2,reg3,reg4], meta_regressor=RandomForestRegressor(n_estimators=best['rfr_meta']['meta_regressor__n_estimators'],max_depth=best['rfr_meta']['meta_regressor__max_depth']))
            fsr.fit(x_train, y_train)
            y_pred = fsr.predict(test_features)
            test_pred = fsr.predict(x_test)
            #print('testing')
            #testing errors
            strmretest2.append(mean_relative_error(y_test,test_pred))
            strmaetest2.append(metrics.mean_absolute_error(y_test,test_pred))
            strrmsetest2.append(math.sqrt(metrics.mean_squared_error(y_test,test_pred)))
            strmsetest2.append(metrics.mean_squared_error(y_test,test_pred))
            strcomptest2.append(measure_completeness(y_test,test_pred))
            strpredltest2.append(predl(y_test,test_pred,0.3))
            strrsqtest2.append(metrics.r2_score(y_test,test_pred))
            #print('validation')
            #validation errors
            strmre2.append(mean_relative_error(test_label,y_pred))
            strmae2.append(metrics.mean_absolute_error(test_label,y_pred))
            strrmse2.append(math.sqrt(metrics.mean_squared_error(test_label,y_pred)))
            strmse2.append(metrics.mean_squared_error(test_label,y_pred))
            strcomp2.append(measure_completeness(test_label,y_pred))
            strpredl2.append(predl(test_label,y_pred,0.3))
            strrsq2.append(metrics.r2_score(test_label,y_pred))
            t12 = time.time()
            total2=t12-t02
            
            i=i+1
        print("Testing Run: ",run+1," ",filename," finished.")

        t1 = time.time()
        total = t1-t0
        dtravgmre.append(np.mean(dtrmre))
        dtravgmae.append(np.mean(dtrmae))
        dtravgrmse.append(np.mean(dtrrmse))
        dtravgmse.append(np.mean(dtrmse))
        dtravgcomp.append(np.mean(dtrcomp))
        dtravgpredl.append(np.mean(dtrpredl))
        dtravgrsq.append(np.mean(dtrrsq))
        dtravgmretest.append(np.mean(dtrmretest))
        dtravgmaetest.append(np.mean(dtrmaetest))
        dtravgrmsetest.append(np.mean(dtrrmsetest))
        dtravgmsetest.append(np.mean(dtrmsetest))
        dtravgcomptest.append(np.mean(dtrcomptest))
        dtravgpredltest.append(np.mean(dtrpredltest))
        dtravgrsqtest.append(np.mean(dtrrsqtest))

        rfravgmre.append(np.mean(rfrmre))
        rfravgmae.append(np.mean(rfrmae))
        rfravgrmse.append(np.mean(rfrrmse))
        rfravgmse.append(np.mean(rfrmse))
        rfravgcomp.append(np.mean(rfrcomp))
        rfravgpredl.append(np.mean(rfrpredl))
        rfravgrsq.append(np.mean(rfrrsq))
        rfravgmretest.append(np.mean(rfrmretest))
        rfravgmaetest.append(np.mean(rfrmaetest))
        rfravgrmsetest.append(np.mean(rfrrmsetest))
        rfravgmsetest.append(np.mean(rfrmsetest))
        rfravgcomptest.append(np.mean(rfrcomptest))
        rfravgpredltest.append(np.mean(rfrpredltest))
        rfravgrsqtest.append(np.mean(rfrrsqtest))

        etravgmre.append(np.mean(etrmre))
        etravgmae.append(np.mean(etrmae))
        etravgrmse.append(np.mean(etrrmse))
        etravgmse.append(np.mean(etrmse))
        etravgcomp.append(np.mean(etrcomp))
        etravgpredl.append(np.mean(etrpredl))
        etravgrsq.append(np.mean(etrrsq))
        etravgmretest.append(np.mean(etrmretest))
        etravgmaetest.append(np.mean(etrmaetest))
        etravgrmsetest.append(np.mean(etrrmsetest))
        etravgmsetest.append(np.mean(etrmsetest))
        etravgcomptest.append(np.mean(etrcomptest))
        etravgpredltest.append(np.mean(etrpredltest))
        etravgrsqtest.append(np.mean(etrrsqtest))

        knnavgmre.append(np.mean(knnmre))
        knnavgmae.append(np.mean(knnmae))
        knnavgrmse.append(np.mean(knnrmse))
        knnavgmse.append(np.mean(knnmse))
        knnavgcomp.append(np.mean(knncomp))
        knnavgpredl.append(np.mean(knnpredl))
        knnavgrsq.append(np.mean(knnrsq))
        knnavgmretest.append(np.mean(knnmretest))
        knnavgmaetest.append(np.mean(knnmaetest))
        knnavgrmsetest.append(np.mean(knnrmsetest))
        knnavgmsetest.append(np.mean(knnmsetest))
        knnavgcomptest.append(np.mean(knncomptest))
        knnavgpredltest.append(np.mean(knnpredltest))
        knnavgrsqtest.append(np.mean(knnrsqtest))

        svravgmre.append(np.mean(svrmre))
        svravgmae.append(np.mean(svrmae))
        svravgrmse.append(np.mean(svrrmse))
        svravgmse.append(np.mean(svrmse))
        svravgcomp.append(np.mean(svrcomp))
        svravgpredl.append(np.mean(svrpredl))
        svravgrsq.append(np.mean(svrrsq))
        svravgmretest.append(np.mean(svrmretest))
        svravgmaetest.append(np.mean(svrmaetest))
        svravgrmsetest.append(np.mean(svrrmsetest))
        svravgmsetest.append(np.mean(svrmsetest))
        svravgcomptest.append(np.mean(svrcomptest))
        svravgpredltest.append(np.mean(svrpredltest))
        svravgrsqtest.append(np.mean(svrrsqtest))

#         bavgmre.append(np.mean(bmre))
#         bavgmae.append(np.mean(bmae))
#         bavgrmse.append(np.mean(brmse))
#         bavgmretest.append(np.mean(bmretest))
#         bavgmaetest.append(np.mean(bmaetest))
#         bavgrmsetest.append(np.mean(brmsetest))

        stravgmre.append(np.mean(strmre))
        stravgmae.append(np.mean(strmae))
        stravgrmse.append(np.mean(strrmse))
        stravgmse.append(np.mean(strmse))
        stravgcomp.append(np.mean(strcomp))
        stravgpredl.append(np.mean(strpredl))
        stravgrsq.append(np.mean(strrsq))
        stravgmretest.append(np.mean(strmretest))
        stravgmaetest.append(np.mean(strmaetest))
        stravgrmsetest.append(np.mean(strrmsetest))
        stravgmsetest.append(np.mean(strmsetest))
        stravgcomptest.append(np.mean(strcomptest))
        stravgpredltest.append(np.mean(strpredltest))
        stravgrsqtest.append(np.mean(strrsqtest))

        stravgmre2.append(np.mean(strmre2))
        stravgmae2.append(np.mean(strmae2))
        stravgrmse2.append(np.mean(strrmse2))
        stravgmse2.append(np.mean(strmse2))
        stravgcomp2.append(np.mean(strcomp2))
        stravgpredl2.append(np.mean(strpredl2))
        stravgrsq2.append(np.mean(strrsq2))
        stravgmretest2.append(np.mean(strmretest2))
        stravgmaetest2.append(np.mean(strmaetest2))
        stravgrmsetest2.append(np.mean(strrmsetest2))
        stravgmsetest2.append(np.mean(strmsetest2))
        stravgcomptest2.append(np.mean(strcomptest2))
        stravgpredltest2.append(np.mean(strpredltest2))
        stravgrsqtest2.append(np.mean(strrsqtest2))
        

#         print('dtr: mre: ', np.mean(dtrmre), ' mae: ', np.mean(dtrmae), ' rmse: ', np.mean(dtrrmse))
#         print('test dtr: mre: ', np.mean(dtrmretest), ' mae: ', np.mean(dtrmaetest), ' rmse: ', np.mean(dtrrmsetest))
#         print('rfr: mre: ', np.mean(rfrmre), ' mae: ', np.mean(rfrmae), ' rmse: ', np.mean(rfrrmse))
#         print('test rfr: mre: ', np.mean(rfrmretest), ' mae: ', np.mean(rfrmaetest), ' rmse: ', np.mean(rfrrmsetest))
#         print('etr: mre: ', np.mean(etrmre), ' mae: ', np.mean(etrmae), ' rmse: ', np.mean(etrrmse))
#         print('test etr: mre: ', np.mean(etrmretest), ' mae: ', np.mean(etrmaetest), ' rmse: ', np.mean(etrrmsetest))
#         print('knn: mre: ', np.mean(knnmre), ' mae: ', np.mean(knnmae), ' rmse: ', np.mean(knnrmse))
#         print('test knn: mre: ', np.mean(knnmretest), ' mae: ', np.mean(knnmaetest), ' rmse: ', np.mean(knnrmsetest))
#         print('svr: mre: ', np.mean(svrmre), ' mae: ', np.mean(svrmae), ' rmse: ', np.mean(svrrmse))
#         print('test svr: mre: ', np.mean(svrmretest), ' mae: ', np.mean(svrmaetest), ' rmse: ', np.mean(svrrmsetest))
# #         print('xgb: mre: ', np.mean(bmre), ' mae: ', np.mean(bmae), ' rmse: ', np.mean(brmse))
# #         print('test xgb: mre: ', np.mean(bmretest), ' mae: ', np.mean(bmaetest), ' rmse: ', np.mean(brmsetest))
#         print('str: mre: ', np.mean(strmre), ' mae: ', np.mean(strmae), ' rmse: ', np.mean(strrmse))
#         print('test str: mre: ', np.mean(strmretest), ' mae: ', np.mean(strmaetest), ' rmse: ', np.mean(strrmsetest))
#         print('str2: mre: ', np.mean(strmre2), ' mae: ', np.mean(strmae2), ' rmse: ', np.mean(strrmse2))
#         print('test str2: mre: ', np.mean(strmretest2), ' mae: ', np.mean(strmaetest2), ' rmse: ', np.mean(strrmsetest2))

#     print('-------------------------------------------------------------------------------------------')
#     print('avg dtr: mre: ', np.mean(dtravgmre), ' mae: ', np.mean(dtravgmae), ' rmse: ', np.mean(dtravgrmse))
#     print('test avg dtr: mre: ', np.mean(dtravgmretest), ' mae: ', np.mean(dtravgmaetest), ' rmse: ', np.mean(dtravgrmsetest))
#     print('avg rfr: mre: ', np.mean(rfravgmre), ' mae: ', np.mean(rfravgmae), ' rmse: ', np.mean(rfravgrmse))
#     print('test avg rfr: mre: ', np.mean(rfravgmretest), ' mae: ', np.mean(rfravgmaetest), ' rmse: ', np.mean(rfravgrmsetest))
#     print('avg etr: mre: ', np.mean(etravgmre), ' mae: ', np.mean(etravgmae), ' rmse: ', np.mean(etravgrmse))
#     print('test avg etr: mre: ', np.mean(etravgmretest), ' mae: ', np.mean(etravgmaetest), ' rmse: ', np.mean(etravgrmsetest))
#     print('avg knn: mre: ', np.mean(knnavgmre), ' mae: ', np.mean(knnavgmae), ' rmse: ', np.mean(knnavgrmse))
#     print('test avg knn: mre: ', np.mean(knnavgmretest), ' mae: ', np.mean(knnavgmaetest), ' rmse: ', np.mean(knnavgrmsetest))
#     print('avg svr: mre: ', np.mean(svravgmre), ' mae: ', np.mean(svravgmae), ' rmse: ', np.mean(svravgrmse))
#     print('test avg svr: mre: ', np.mean(svravgmretest), ' mae: ', np.mean(svravgmaetest), ' rmse: ', np.mean(svravgrmsetest))
# #     print('avg xgb: mre: ', np.mean(bavgmre), ' mae: ', np.mean(bavgmae), ' rmse: ', np.mean(bavgrmse))
# #     print('test avg xgb: mre: ', np.mean(bavgmretest), ' mae: ', np.mean(bavgmaetest), ' rmse: ', np.mean(bavgrmsetest))
#     print('avg str: mre: ', np.mean(stravgmre), ' mae: ', np.mean(stravgmae), ' rmse: ', np.mean(stravgrmse))
#     print('test avg str: mre: ', np.mean(stravgmretest), ' mae: ', np.mean(stravgmaetest), ' rmse: ', np.mean(stravgrmsetest))
#     print('avg str2: mre: ', np.mean(stravgmre2), ' mae: ', np.mean(stravgmae2), ' rmse: ', np.mean(stravgrmse2))
#     print('test avg str2: mre: ', np.mean(stravgmretest2), ' mae: ', np.mean(stravgmaetest2), ' rmse: ', np.mean(stravgrmsetest2))

    with open('resultswc-'+ os.path.splitext(filename)[0] +'.txt', 'w') as f:
                        f.write("testing results:\n")
                        f.write("mre, mae, rmse, mse, comp, pred0.3, rsq\n")
                        f.write("dtr: %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s)\n"% (np.mean(dtravgmre),np.std(dtravgmre),
                                                                            np.mean(dtravgmae), np.std(dtravgmae),
                                                                            np.mean(dtravgrmse), np.std(dtravgrmse),
                                                                            np.mean(dtravgmse), np.std(dtravgmse),
                                                                            np.mean(dtravgcomp), np.std(dtravgcomp),
                                                                            np.mean(dtravgpredl), np.std(dtravgpredl),
                                                                            np.mean(dtravgrsq), np.std(dtravgrsq)))
                        f.write("rfr: %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s)\n"% (np.mean(rfravgmre),np.std(rfravgmre),
                                                                            np.mean(rfravgmae), np.std(rfravgmae),
                                                                            np.mean(rfravgrmse), np.std(rfravgrmse),
                                                                            np.mean(rfravgmse), np.std(rfravgmse),
                                                                            np.mean(rfravgcomp), np.std(rfravgcomp),
                                                                            np.mean(rfravgpredl), np.std(rfravgpredl),
                                                                            np.mean(rfravgrsq), np.std(rfravgrsq)))
                        f.write("etr: %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s)\n"% (np.mean(etravgmre),np.std(etravgmre),
                                                                            np.mean(etravgmae), np.std(etravgmae),
                                                                            np.mean(etravgrmse), np.std(etravgrmse),
                                                                            np.mean(etravgmse), np.std(etravgmse),
                                                                            np.mean(etravgcomp), np.std(etravgcomp),
                                                                            np.mean(etravgpredl), np.std(etravgpredl),
                                                                            np.mean(etravgrsq), np.std(etravgrsq)))
                        f.write("knn: %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s)\n"% (np.mean(knnavgmre),np.std(knnavgmre),
                                                                            np.mean(knnavgmae), np.std(knnavgmae),
                                                                            np.mean(knnavgrmse), np.std(knnavgrmse),
                                                                            np.mean(knnavgmse), np.std(knnavgmse),
                                                                            np.mean(knnavgcomp), np.std(knnavgcomp),
                                                                            np.mean(knnavgpredl), np.std(knnavgpredl),
                                                                            np.mean(knnavgrsq), np.std(knnavgrsq)))
                        f.write("svr: %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s)\n"% (np.mean(svravgmre),np.std(svravgmre),
                                                                            np.mean(svravgmae), np.std(svravgmae),
                                                                            np.mean(svravgrmse), np.std(svravgrmse),
                                                                            np.mean(svravgmse), np.std(svravgmse),
                                                                            np.mean(svravgcomp), np.std(svravgcomp),
                                                                            np.mean(svravgpredl), np.std(svravgpredl),
                                                                            np.mean(svravgrsq), np.std(svravgrsq)))
#                         f.write("xgb: %s(%s) & %s(%s) & %s(%s)\n"% (np.mean(bavgmre),np.std(bavgmre),
#                                                                             np.mean(bavgmae), np.std(bavgmae),
#                                                                             np.mean(bavgrmse), np.std(bavgrmse)))
                        f.write("str_1: %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s)\n"% (np.mean(stravgmre),np.std(stravgmre),
                                                                            np.mean(stravgmae), np.std(stravgmae),
                                                                            np.mean(stravgrmse), np.std(stravgrmse),
                                                                            np.mean(stravgmse), np.std(stravgmse),
                                                                            np.mean(stravgcomp), np.std(stravgcomp),
                                                                            np.mean(stravgpredl), np.std(stravgpredl),
                                                                            np.mean(stravgrsq), np.std(stravgrsq)))
                        f.write("str_2: %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s)\n"% (np.mean(stravgmre2),np.std(stravgmre2),
                                                                            np.mean(stravgmae2), np.std(stravgmae2),
                                                                            np.mean(stravgrmse2), np.std(stravgrmse2),
                                                                            np.mean(stravgmse2), np.std(stravgmse2),
                                                                            np.mean(stravgcomp2), np.std(stravgcomp2),
                                                                            np.mean(stravgpredl2), np.std(stravgpredl2),
                                                                            np.mean(stravgrsq2), np.std(stravgrsq2)))
                        f.write("validation results:\n")
                        f.write("mre, mae, rmse, mse, comp, pred0.3, rsq\n")
                        f.write("dtr: %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s)\n"% (np.mean(dtravgmretest),np.std(dtravgmretest),
                                                                            np.mean(dtravgmaetest), np.std(dtravgmaetest),
                                                                            np.mean(dtravgrmsetest), np.std(dtravgrmsetest),
                                                                            np.mean(dtravgmsetest), np.std(dtravgmsetest),
                                                                            np.mean(dtravgcomptest), np.std(dtravgcomptest),
                                                                            np.mean(dtravgpredltest), np.std(dtravgpredltest),
                                                                            np.mean(dtravgrsqtest), np.std(dtravgrsqtest)))
                        f.write("rfr: %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s)\n"% (np.mean(rfravgmretest),np.std(rfravgmretest),
                                                                            np.mean(rfravgmaetest), np.std(rfravgmaetest),
                                                                            np.mean(rfravgrmsetest), np.std(rfravgrmsetest),
                                                                            np.mean(rfravgmsetest), np.std(rfravgmsetest),
                                                                            np.mean(rfravgcomptest), np.std(rfravgcomptest),
                                                                            np.mean(rfravgpredltest), np.std(rfravgpredltest),
                                                                            np.mean(rfravgrsqtest), np.std(rfravgrsqtest)))
                        f.write("etr: %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s)\n"% (np.mean(etravgmretest),np.std(etravgmretest),
                                                                            np.mean(etravgmaetest), np.std(etravgmaetest),
                                                                            np.mean(etravgrmsetest), np.std(etravgrmsetest),
                                                                            np.mean(etravgmsetest), np.std(etravgmsetest),
                                                                            np.mean(etravgcomptest), np.std(etravgcomptest),
                                                                            np.mean(etravgpredltest), np.std(etravgpredltest),
                                                                            np.mean(etravgrsqtest), np.std(etravgrsqtest)))
                        f.write("knn: %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s)\n"% (np.mean(knnavgmretest),np.std(knnavgmretest),
                                                                            np.mean(knnavgmaetest), np.std(knnavgmaetest),
                                                                            np.mean(knnavgrmsetest), np.std(knnavgrmsetest),
                                                                            np.mean(knnavgmsetest), np.std(knnavgmsetest),
                                                                            np.mean(knnavgcomptest), np.std(knnavgcomptest),
                                                                            np.mean(knnavgpredltest), np.std(knnavgpredltest),
                                                                            np.mean(knnavgrsqtest), np.std(knnavgrsqtest)))
                        f.write("svr: %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s)\n"% (np.mean(svravgmretest),np.std(svravgmretest),
                                                                            np.mean(svravgmaetest), np.std(svravgmaetest),
                                                                            np.mean(svravgrmsetest), np.std(svravgrmsetest),
                                                                            np.mean(svravgmsetest), np.std(svravgmsetest),
                                                                            np.mean(svravgcomptest), np.std(svravgcomptest),
                                                                            np.mean(svravgpredltest), np.std(svravgpredltest),
                                                                            np.mean(svravgrsqtest), np.std(svravgrsqtest)))
#                         f.write("xgb: %s(%s) & %s(%s) & %s(%s)\n"% (np.mean(bavgmretest),np.std(bavgmretest),
#                                                                             np.mean(bavgmaetest), np.std(bavgmaetest),
#                                                                             np.mean(bavgrmsetest), np.std(bavgrmsetest)))
                        f.write("str_1: %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s)\n"% (np.mean(stravgmretest),np.std(stravgmretest),
                                                                            np.mean(stravgmaetest), np.std(stravgmaetest),
                                                                            np.mean(stravgrmsetest), np.std(stravgrmsetest),
                                                                            np.mean(stravgmsetest), np.std(stravgmsetest),
                                                                            np.mean(stravgcomptest), np.std(stravgcomptest),
                                                                            np.mean(stravgpredltest), np.std(stravgpredltest),
                                                                            np.mean(stravgrsqtest), np.std(stravgrsqtest)))
                        f.write("str_1 time: %s\n"% (total1))
                        f.write("str_2: %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s) & %s(%s)\n"% (np.mean(stravgmretest2),np.std(stravgmretest2),
                                                                            np.mean(stravgmaetest2), np.std(stravgmaetest2),
                                                                            np.mean(stravgrmsetest2), np.std(stravgrmsetest2),
                                                                            np.mean(stravgmsetest2), np.std(stravgmsetest2),
                                                                            np.mean(stravgcomptest2), np.std(stravgcomptest2),
                                                                            np.mean(stravgpredltest2), np.std(stravgpredltest2),
                                                                            np.mean(stravgrsqtest2), np.std(stravgrsqtest2)))
                        f.write("str_2 time: %s\n"% (total2))
                        f.write("Total time: %s\n"% (total))
                        
folders=['ant','camel','ivy','jedit','log4j','lucene','poi','velocity','xalan','xerces']
def run_cross(index):
    train_folders=[]
    test_index=index
    print(folders[index])
    if index==0:
        train_folders=folders[1:]
    elif index==len(folders)-1:
        train_folders=folders[:-1]
    else:
        train_folders=folders[:index]
        train_folders.extend(folders[index+1:])
    run_techniques(train_folders,[folders[test_index]])
    
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(run_cross, i) for i in range(len(folders))]
    results = [future.result() for future in concurrent.futures.as_completed(futures)]