# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:10:54 2020

@author: JMGC2008
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

def randomForest(data,colN):
    print(data)
    y=data["class"]
    listDrop=['class']
    X=data.drop(listDrop,axis="columns")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    nEst=[]
    for i in range(1,200,3):
        nEst.append(i)
    acc=[]
    oob_error=[]
    for i in nEst:
        clf_p = RandomForestClassifier(n_estimators=i, warm_start=True, oob_score=True, random_state=1)
        clf_p.fit(X_train, y_train)
        y_pred_p = clf_p.predict(X_test)
        score = accuracy_score(y_test, y_pred_p)
        acc.append(score)
        oob_error.append(1-clf_p.oob_score_)
        
    
    plt.figure(1)
    plt.plot(nEst,oob_error)
    plt.xlabel("number of trees")
    plt.ylabel("oob")
    plt.legend()
    index_max = max(range(len(acc)), key=acc.__getitem__)
    print(nEst[index_max])
    plt.figure(2)
    plotFeature(clf_p,colN)
        
def plotFeature(model,colN):
    for feature in zip(colN, model.feature_importances_):
        print (feature)
    sortedIdx=model.feature_importances_.argsort()
    print(sortedIdx)
    print(model.feature_importances_)
    colN=np.array(colN)
    plt.barh(colN[sortedIdx], model.feature_importances_[sortedIdx])

def load():
    
    colN=['clump_thickness','unif_cell_size','unif_cell_shape', 'marg_adhesion', 'single_epith_cell_size',
                                                       'bare_nuclei', 'bland_chromatin', 'normal_nucleoli','mitoses']
    
    data=pd.read_csv("breastCancerImp.csv")
    
    randomForest(data,colN)

def main():
    load()

if __name__=="__main__":
    main()