# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:22:43 2020

@author: JMGC2008
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing

def normalization(data,colN):
    colNames=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","Age"]
    data2=data[colNames]
    min_max_scaler=preprocessing.MinMaxScaler()
    x_scaled=min_max_scaler.fit_transform(data2)
    data[colNames]=x_scaled
    return data

def load():
    colNames=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Class"]
    data=pd.read_csv("pima-indians-diabetes.csv",names=colNames)
    
    
    data2=normalization(data,colNames)
    
    
    
    train_val, test = train_test_split(data2, test_size=0.1, random_state=42)
    train, validate = train_test_split(train_val, test_size=0.2, random_state=42)
    
    yTrain=train["Class"]
    
    XTrain=train.drop("Class",axis="columns")
    
    
    XTest=test.drop("Class",axis="columns")
    
   
    yTest=test["Class"]
    
    XVal=validate.drop("Class",axis="columns")
    yVal=validate["Class"]
    #fit train set
    accTrain=[]
    accVal=[]
    maxTrees=[]
    for i in range(1,200):
        model = AdaBoostClassifier(n_estimators=i, learning_rate=1, algorithm="SAMME")
        model.fit(XTrain, yTrain)
        pred = model.predict(XTrain)
        acc = accuracy_score(yTrain, pred)
        accTrain.append(acc)
        test = model.predict(XTest)
        acc = accuracy_score(yTest, test)
        accVal.append(acc)
        maxTrees.append(i)
    
    print("train max accuracy",max(accTrain))
    index_max = max(range(len(accTrain)), key=accTrain.__getitem__)
    
    print("val max accuracy ",max(accVal))
    index_max = max(range(len(accVal)), key=accVal.__getitem__)
    print("number of trees by validation set ",maxTrees[index_max])
    plt.figure(1)
    axes = plt.gca()
    axes.set_xlim([1,200])
    axes.set_ylim([0.6,1.0])
    d=pd.DataFrame({"accTrain":pd.Series(accTrain),"accVal":pd.Series(accVal),"max_trees":pd.Series(maxTrees)})
    
    
    plt.plot("max_trees","accTrain",data=d,label="train")
    plt.plot("max_trees","accVal",data=d,label="val")
    plt.xlabel("max trees")
    plt.ylabel("accuracy")
    plt.legend()
    
    
    #use validation set
    learningR=np.linspace(0.001,1.5,10)
    learning=[]
    accTrain=[]
    accVal=[]
    
    for i in learningR:
        model = AdaBoostClassifier(n_estimators=15, learning_rate=i, algorithm="SAMME")
        model.fit(XTrain, yTrain)
        pred = model.predict(XTrain)
        acc = accuracy_score(yTrain, pred)
        accTrain.append(acc)
        val = model.predict(XVal)
        acc = accuracy_score(yVal, val)
        accVal.append(acc)
        learning.append(i)
        
    print("validation")
    print("train ",max(accTrain))
    index_max = max(range(len(accTrain)), key=accTrain.__getitem__)
    print("index max value ",index_max)
    print("max value ",max(accVal))
    index_max = max(range(len(accVal)), key=accVal.__getitem__)
    print("learning rate value ",accVal[index_max])
    
    plt.figure(2)
    axes = plt.gca()
    axes.set_xlim([0,1.5])
    axes.set_ylim([0.6,1.0])
    d=pd.DataFrame({"accTrain":pd.Series(accTrain),"accVal":pd.Series(accVal),"learningRate":pd.Series(learning)})
    #plt.plot("learningRate","accTrain","accVal",data=d,label="train")
    plt.plot("learningRate","accTrain",data=d,label="train")
    plt.plot("learningRate","accVal",data=d,label="val")
    plt.xlabel("learning Rate")
    plt.ylabel("accuracy")
    plt.legend()
    #use test set
    print("test")
    model=AdaBoostClassifier(n_estimators=15,learning_rate=0.80,algorithm="SAMME")
    print(model)
    
    print("cross val score on train",cross_val_score(model, XTrain, yTrain, cv = 10).mean())
    
    model.fit(XTrain,yTrain)
    testPred=model.predict(XTest)
    acc=accuracy_score(yTest,testPred)
    print (acc)
    
    
    

def main():
    load()


if __name__=="__main__":
    main()























