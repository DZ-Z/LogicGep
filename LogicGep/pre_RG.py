from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.feature_selection import mutual_info_classif
from xgboost import plot_importance
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
def data_pre (target,a): 
    data_y = a.loc[ : ,target]
    data_x=a.drop([target],axis=1)
    data_1=data_x.drop(data_x.tail(1).index)
    data_2=data_y.drop(data_y.head(1).index,)
    list1=[]
    for i in range(0,len(data_1)):
        list1.append(i)
    data_1.index =list1
    data_2.index=list1
    zz=pd.concat([data_1,data_2],axis=1)
    return zz

def RF_XG_regulators(x,y,zz):
    col = zz.columns
    rfc = RandomForestClassifier(n_estimators=1000, min_samples_leaf=1, n_jobs=-1)
    rfc.fit(x,y) 
    importance_rfc = rfc.feature_importances_
    re_rfc = pd.DataFrame({'feature':np.array(col)[:-1],'IMP':importance_rfc}).sort_values(by = 'IMP',axis = 0,ascending = False)
    re_rfc=re_rfc.head(10)
    RF_features=re_rfc["feature"].tolist()
    le = LabelEncoder()
    y_train = le.fit_transform(y)
    model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='binary:logistic',
                              n_jobs=-1)
    model.fit(x, y_train)
    importance_xgb = model.feature_importances_
    re_xgb = pd.DataFrame({'feature':np.array(col)[:-1],'IMP':importance_xgb}).sort_values(by = 'IMP',axis = 0,ascending = False)
    re_xgb=re_xgb.head(10)
    XG_features=re_xgb["feature"].tolist()
    return RF_features，XG_features

def Regulators(target,a):
    ab=a
    regu=[]
    zz=data_pre(target,ab)
    x,y = np.split(zz, (len(zz.columns)-1,), axis = 1)  
    RF_features,XG_features=RF_XG_regulators(x,y,zz)
    RG_sets =list(set(RF_features).union(set(XG_features)))
  
    return RG_sets



























