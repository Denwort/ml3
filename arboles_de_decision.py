# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn import preprocessing, model_selection, linear_model
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
from pandas.plotting import scatter_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image  
import pydotplus       #conda install -c anaconda pydotplus
            


def load():
    data2=pd.read_csv("breast-cancer-wisconsin.data",names=['id', 'clump_thickness','unif_cell_size',
                                                                           'unif_cell_shape', 'marg_adhesion', 'single_epith_cell_size',
                                                                           'bare_nuclei', 'bland_chromatin', 'normal_nucleoli','mitoses','class'])
    decisionTree(data2)   
    

def decisionTree(data):
    listDrop=['id']
    data=data.drop(listDrop,axis="columns")
    
    data.replace('?',np.nan,inplace=True)
    imp=SimpleImputer(missing_values=np.NaN,strategy='mean')
    data=pd.DataFrame(imp.fit_transform(data))
    
    columnNames=['clump_thickness','unif_cell_size','unif_cell_shape', 'marg_adhesion', 'single_epith_cell_size','bare_nuclei', 'bland_chromatin', 'normal_nucleoli','mitoses','class']
    data.columns=columnNames

    #completar    
    
    train_val,test=train_test_split(data, test_size=0.1,random_state=42)
    train,validate=train_test_split(train_val, test_size=0.2,random_state=42)
    
    yTrain=train["class"]
    XTrain=train.drop("class",axis="columns")
    
    yTest=test["class"]
    XTest=test.drop("class",axis="columns")
    
    accGini=[]
    accEntropy=[]
    maxDepth=[]
    
    accEntTrain=[]
    
    for i in range(1,40):
        model=DecisionTreeClassifier(criterion="entropy",max_features="log2",max_depth=i,random_state=2)
        model.fit(XTrain,yTrain)
        
        
        # Graficar curva
        predT=model.predict(XTrain)
        acc=accuracy_score(yTrain,predT)
        accEntTrain.append(acc)
        
        pred=model.predict(XTest)
        acc=accuracy_score(yTest,pred)
        accEntropy.append(acc)
        
        model=DecisionTreeClassifier(criterion="gini",max_features="log2",max_depth=i,random_state=2)
        model.fit(XTrain,yTrain)
        pred=model.predict(XTest)
        acc=accuracy_score(yTest,pred)
        accGini.append(acc)
        
        
        maxDepth.append(i)
   
    
    print(accEntropy)
    print("train data")
    print(accEntTrain)
    
    
    axes = plt.gca()
    axes.set_xlim([1,40])
    axes.set_ylim([0.80,1.05])
    
    
    d=pd.DataFrame({"acc_entropyT":pd.Series(accEntTrain),"acc_entropy":pd.Series(accEntropy),"max_depth":pd.Series(maxDepth)})
    
    plt.plot("max_depth","acc_entropy",'b',data=d,label="entropy")
    plt.plot("max_depth","acc_entropyT",'r',data=d,label="entropy_train")
    plt.xlabel("max_depth")
    plt.ylabel("accuracy")
    plt.legend()
        
    dt=DecisionTreeClassifier(criterion="entropy",max_features="log2",max_depth=5,random_state=2)
    dt.fit(XTrain,yTrain)
    pred=model.predict(XTest)
    acc=accuracy_score(yTest,pred)
    print(acc)
    print(dt.tree_.threshold)
    
    names=['clump_thickness','unif_cell_size','unif_cell_shape', 'marg_adhesion', 'single_epith_cell_size','bare_nuclei', 'bland_chromatin', 'normal_nucleoli','mitoses']
    dot_data = StringIO()
    export_graphviz(dt, out_file=dot_data,feature_names=names,class_names=['2','4'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('tree.png')
    Image(graph.create_png()) 
    


def main():
    load()

if __name__=="__main__":
    main()