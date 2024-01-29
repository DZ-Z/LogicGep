from sklearn.ensemble import RandomForestClassifier
import pandas as pd
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

def RFregulators(x,y,zz):
    col = zz.columns
#   randomforests
    rfc = RandomForestClassifier(n_jobs=-1)
    rfc.fit(x,y) 
    importance_rfc = rfc.feature_importances_
    re_rfc = pd.DataFrame({'feature':np.array(col)[:-1],'IMP':importance_rfc}).sort_values(by = 'IMP',axis = 0,ascending = False)
    re_rfc=re_rfc.head(10)
    RF_features=re_rfc["feature"].tolist()
    return RF_features

def Regulators(target,a):
    ab=a
    regu=[]
    zz=data_pre(target,ab)
    x,y = np.split(zz, (len(zz.columns)-1,), axis = 1)
    RF_features= RFregulators(x, y, zz)
    return RF_features




























