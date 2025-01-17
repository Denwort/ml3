
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.io.arff import loadarff
from matplotlib.lines import Line2D
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, cross_val_score,GridSearchCV
from sklearn.model_selection import KFold,LeaveOneOut,StratifiedKFold
from scipy import stats
import statsmodels.api as sm
from sklearn.feature_selection import RFE

#logistic

import matplotlib
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn import preprocessing, model_selection, linear_model
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_predict
from pandas.plotting import scatter_matrix
from sklearn.metrics import accuracy_score,make_scorer, f1_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


from imblearn.pipeline import Pipeline
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from collections import OrderedDict

# Configuracion de pandas para imprimir todo el data set
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_colwidth', None)

# Cargar el csv
def load():
  df = pd.read_csv("yeast.data", sep='\s+', header=None)
  columns = ["Sequence_Name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "localization_site"]
  df.columns = columns
  # Dropear ID
  df = df.iloc[:, df.columns != 'Sequence_Name']
  return df

# Analisis exploratorio

# Graficar una variables
def plotTarget(data, target):
  categoria_counts = data[target].value_counts()
  porcentajes = data[target].value_counts(normalize=True) * 100
  print(f"Cat\tCant\tPorc")
  for categoria, cantidad, porcentaje in zip(categoria_counts.index, categoria_counts.values, porcentajes.values):
    print(f"{categoria}\t{cantidad}\t{porcentaje:.2f}%")
  categorias_ordenadas = categoria_counts.index
  plt.figure(figsize=(8, 6))
  plt.bar(categorias_ordenadas, categoria_counts, color='skyblue')
  plt.xlabel('Categoría')
  plt.ylabel('Frecuencia')
  plt.title('Distribución de Categorías')
  plt.show()

# Histograma y diagrama de caja para variables numericas
def analisisNumericas(df):
  varNumericas = df.select_dtypes(include=np.number).columns
  print("Numericas: ", varNumericas)
  for numerica in varNumericas:
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    df[numerica].plot.hist(bins=25, ax=axes[0])
    axes[0].set_title(f'Histograma de {numerica}', fontdict={'fontsize': 16})
    df[numerica].plot.box(ax=axes[1])
    axes[1].set_title(f'Boxplot de {numerica}', fontdict={'fontsize': 16})
    plt.show()
    print("\n")

# Matriz de correlacion
def correlacion(df, target):
  df = df.drop(columns=[target])
  correlation_matrix = df.corr()
  print(correlation_matrix)
  plt.figure(figsize=(10, 8))
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
  plt.title('Matriz de Correlación')
  plt.show()

# Analisis de nulos
def nullAnalysis(df):
  null_columns=df.isnull().any()
  print("Nulos en columnas:")
  print(null_columns)
  null_sum = df.isnull().sum()
  print("Suma de nulos:")
  print(null_sum)

# Encoding
#  Encoding con LabelEncoder
def encodingLabel(df, target, mapping):
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(list(mapping.keys()))
    df[target] = label_encoder.transform(df[target])
    return df

# Tratamiento de Outliers
#  Algoritmo LOF
def lof(X, contamination, plot):
    lof=LocalOutlierFactor(n_neighbors=3,contamination=contamination)
    y_pred=lof.fit_predict(X)
    novelty_scores=-lof.negative_outlier_factor_
    threshold=np.percentile(novelty_scores, (1 - contamination) * 100)
    predicted_labels=np.where(y_pred==-1,1,0)
    anomaly_indices=np.where(y_pred==-1)[0]
    #print("indices de las anomalias")
    #print(anomaly_indices)
    #print("datos clasificados como anomalias")
    #print(df.iloc[anomaly_indices])
    if plot:
      plotLOF(X.iloc[:,:].values, novelty_scores, threshold, predicted_labels, y_pred)
    return anomaly_indices 

#  Plotear resultados del LOF
def plotLOF(X,novelty_scores,threshold,predicted_labels,y_pred):

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(novelty_scores, bins=20, color='skyblue', edgecolor='black')
    plt.title("Histogram of Novelty Scores")
    plt.xlabel("Novelty Score")
    plt.ylabel("Frequency")
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    colors = np.array(['red', 'blue'])
    plt.scatter(X[:, 0], X[:, 1], c=colors[(y_pred + 1) // 2], s=50, edgecolors='k')
    plt.title("Local Outlier Factor (LOF)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(['Normal', 'Outlier'], loc='best')
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Normal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Outlier')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.show()

#  Descartar outliers segun LOF
def tratamientoOutliers(df, target, contamination, plot):
  X = df.drop(target, axis=1)
  anomalias=lof(X, contamination, plot)
  df = df.drop(anomalias)
  df = df.reset_index(drop=True)
  return df

# Modelos

def agregar_modelo(pipeline, classifier):
    model = clone(pipeline)
    model.steps.append(('classifier', classifier))
    return model

def get_score(modelo, X_test, y_test, plot=False):
    y_pred = modelo.predict(X_test)
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("f1 score: ", f1_score(y_test, y_pred, average='weighted'))
    if plot == True:
      conf_matrix = confusion_matrix(y_test, y_pred)
      plt.figure(figsize=(8, 6))
      mapping = {'CYT': 0, 'NUC': 1, 'MIT': 2, 'ME3': 3, 'ME2': 4, 'ME1': 5, 'EXC': 6, 'VAC': 7, 'POX': 8, 'ERL': 9}
      sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False,
                  xticklabels=mapping,
                  yticklabels=mapping)
      plt.xlabel('Predicción')
      plt.ylabel('Etiqueta Real')
      plt.title('Matriz de Confusión')
      plt.show()

def nested_cv(pipeline, gs_function, X, y):
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=123)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=123)
    outer_scores = []
    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        best_model = gs_function(pipeline, X_train, y_train, inner_cv)
        y_pred = best_model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        outer_scores.append(score)
        print(f"Nested CV Accuracy: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")

#  Decision tree
def decisiontreeGS(pipeline, X_train, y_train, cv=5):
  param_grid = {
      'classifier__criterion': ['gini', 'entropy'],
      'classifier__max_depth': [None, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
  }
  grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='f1_weighted', verbose=1, n_jobs=-1)
  grid_search.fit(X_train, y_train)
  print("Mejores hiperparámetros:")
  print(grid_search.best_params_)
  
  return grid_search.best_estimator_


def decisionTreePlot(XTrain, yTrain, XVal, yVal):
  accEntropyTrain=[]
  accEntropyVal=[]
  accGiniTrain=[]
  accGiniVal=[]
  maxDepth=[]
  for i in range(1,40):
      # Entropy
      model=DecisionTreeClassifier(criterion="entropy",max_features="log2",max_depth=i,random_state=2)
      model.fit(XTrain,yTrain)
      #  Train
      pred=model.predict(XTrain)
      acc=f1_score(yTrain,pred, average='weighted')
      accEntropyTrain.append(acc)
      #  Validate
      pred=model.predict(XVal)
      acc=f1_score(yVal,pred, average='weighted')
      accEntropyVal.append(acc)
      # Gini
      model=DecisionTreeClassifier(criterion="gini",max_features="log2",max_depth=i,random_state=2)
      model.fit(XTrain,yTrain)
      # Train
      pred=model.predict(XTrain)
      acc=f1_score(yTrain,pred, average='weighted')
      accGiniTrain.append(acc)
      #  Validate
      pred=model.predict(XVal)
      acc=f1_score(yVal,pred, average='weighted')
      accGiniVal.append(acc)
    
      maxDepth.append(i)
  axes = plt.gca()
  axes.set_xlim([1,40])
  axes.set_ylim([0.80,1.05])
  dataEntropy=pd.DataFrame({"acc_entropyT":pd.Series(accEntropyTrain),"acc_entropy":pd.Series(accEntropyVal),"max_depth":pd.Series(maxDepth)})
  dataGini=pd.DataFrame({"acc_giniT":pd.Series(accGiniTrain),"acc_gini":pd.Series(accGiniVal),"max_depth":pd.Series(maxDepth)})
  plt.plot("max_depth","acc_entropy",'b',data=dataEntropy,label="entropy_test")
  plt.plot("max_depth","acc_entropyT",'r',data=dataEntropy,label="entropy_train")
  plt.plot("max_depth","acc_gini",'g',data=dataGini,label="gini_test")
  plt.plot("max_depth","acc_giniT",'y',data=dataGini,label="gini_train")
  plt.xlabel("max_depth")
  plt.ylabel("f1 score")
  plt.legend()
  plt.show()

#  AdaBoost
def adaboostGS(pipeline, X_train, y_train, cv=5):
  param_grid = {
      'classifier__n_estimators': [10, 100, 200, 1000],
      'classifier__learning_rate': [0.001, 0.005, .01, 0.05, 0.1, 0.5, 1, 5, 10]
  }
  grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='f1_weighted', verbose=1, n_jobs=-1)
  grid_search.fit(X_train, y_train)
  print("Mejores hiperparámetros:")
  print(grid_search.best_params_)
  return grid_search.best_estimator_

def adaboostPlot(XTrain, yTrain, XVal, yVal):
  learningR=np.linspace(0.001,1.5,10)
  learning=[]
  accTrain=[]
  accVal=[]
  
  for i in range(1,200):
    model = AdaBoostClassifier(n_estimators=i, learning_rate=1, algorithm="SAMME")
    model.fit(XTrain, yTrain)
    pred = model.predict(XTrain)
    acc = f1_score(yTrain, pred, average='weighted')
    accTrain.append(acc)
    val = model.predict(XVal)
    acc = f1_score(yVal, val, average='weighted')
    accVal.append(acc)
    learning.append(i)
  
  plt.figure(2)
  axes = plt.gca()
  axes.set_xlim([0,1.5])
  axes.set_ylim([0.6,1.0])
  d=pd.DataFrame({"accTrain":pd.Series(accTrain),"accVal":pd.Series(accVal),"learningRate":pd.Series(learning)})
  plt.plot("learningRate","accTrain",data=d,label="train")
  plt.plot("learningRate","accVal",data=d,label="val")
  plt.xlabel("n_estimators")
  plt.ylabel("f1 score")
  plt.legend()
  plt.show()
   
# Random forest
def randomforestSinAfinamiento(pipeline, X_train, y_train):
  pipeline.fit(X_train, y_train)
  return pipeline

def randomforestRGS(pipeline, X_train, y_train):
  param_distributions = {
      'classifier__n_estimators': [50, 100, 200, 500, 1000],
      'classifier__max_features': [None, 'sqrt', 'log2'],
      'classifier__max_depth': [None, 1, 5, 10, 20, 40],
  }

  random_search = RandomizedSearchCV(
      estimator=pipeline,
      param_distributions=param_distributions,
      n_iter=50,
      cv=5,
      scoring='f1_weighted',
      verbose=1,
      n_jobs=-1,
      random_state=123
  )
  
  random_search.fit(X_train, y_train)
  print("Mejores hiperparámetros:")
  print(random_search.best_params_)
  
  return random_search.best_estimator_


def plotRandomForest(X_train, y_train, X_test, y_test):
    nEst=[]
    for i in range(1,200,3):
        nEst.append(i)
    max_depth = 10  # Definir la profundidad máxima de los árboles
    max_features = 'sqrt'  # Número de características a considerar para cada división
    random_state = 123  # Semilla para reproducibilidad
    f1_scores = []
    for n_estimators in nEst:
        # Crear el clasificador con los parámetros actuales
        rf_clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            bootstrap=True,
            oob_score=True,
            random_state=random_state
        )
        rf_clf.fit(X_train, y_train)
        y_pred = rf_clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_scores.append(f1)
    plt.figure(figsize=(10, 6))
    plt.plot(nEst, f1_scores, marker='o', linestyle='-', color='b')
    plt.xlabel('n_estimators')
    plt.ylabel('F1 score')
    plt.grid(True)
    plt.show()

# Feature selection
#  Algoritmo forward selection
def forward_selection(X, y, threshold=0.01):
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    selected_features = []
    remaining_features = list(X.columns)
    best_score = -np.inf
    last_best_score = -np.inf

    while remaining_features:
        scores = []
        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_subset = X[current_features]
            model = LogisticRegression(random_state=123)
            score = np.mean(cross_val_score(model, X_subset, y, cv=5, scoring='accuracy'))
            scores.append((score, feature))
        
        scores.sort(reverse=True, key=lambda x: x[0])
        best_score, best_feature = scores[0]

        # Check if the improvement is greater than the threshold
        if best_score - last_best_score > threshold:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            last_best_score = best_score
            
        else:
            break
    print(f"Forward selection: {selected_features}, Score: {best_score}")
    return selected_features

#  Algoritmo Recursive Feature Elimination
def recursive_feature_elimination(X,y, n_features):
  scaler = StandardScaler()
  X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
  model=LogisticRegression()
  rfe=RFE(model,n_features_to_select=n_features)
  fit=rfe.fit(X, y)
  print("Recursive forward elimination: ",X.columns[fit.support_])
  return X.columns[fit.support_]

# Feature selection con Random Forest
def rf_features(X, y, n_estimators=100):
  scaler = StandardScaler()
  X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
  rf = RandomForestClassifier(n_estimators=n_estimators, random_state=123, oob_score=True)
  rf.fit(X, y)
  importances = rf.feature_importances_
  feature_names = X.columns
  feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
  feature_importances = feature_importances.sort_values(by='importance', ascending=False)
  threshold = np.median(importances)
  important_features = feature_importances[feature_importances['importance'] > threshold]['feature']
  print("Random forest features:\n", important_features)
  return important_features

def main():
    
    df = load()

    # Analisis exploratorio
    #analisisNumericas(df)
    #plotTarget(df, 'localization_site')
    #correlacion(df, 'localization_site')
    
    # Analisis de nulos
    #nullAnalysis(df)
    
    # Aplicar feature selection
    #df = df[['localization_site', 'alm', 'nuc', 'mit', 'mcg', 'gvh']] 

    # Tratamiento de outliers
    df = tratamientoOutliers(df, 'localization_site', contamination=0.07, plot=False)

    # Encoding
    mapping = {'CYT': 0,'NUC': 1,'MIT': 2,'ME3': 3,'ME2': 4,'ME1': 5,'EXC': 6,'VAC': 7,'POX': 8,'ERL': 9}
    df = encodingLabel(df, 'localization_site', mapping)

    # Dividir dataset
    X = df.drop('localization_site',axis=1)
    y = df['localization_site']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)

    # Feature selection
    #forward_selection(X, y, 0.01)
    #recursive_feature_elimination(X, y, 5)
    #rf_features(X, y, 100)

    # Pipeline: Balanceo, Escalamiento
    pipeline = Pipeline([
        #('smote', SMOTE(k_neighbors=2, random_state=123)), # Balanceo
        #('borderline_smote', BorderlineSMOTE(k_neighbors=2,random_state=123)), #Balanceo
        #('smotetomek', SMOTETomek(smote = SMOTE(k_neighbors=2, random_state=123), random_state=123)), #Balanceo
        #('smoteenn', SMOTEENN(smote = SMOTE(k_neighbors=2, random_state=123),random_state=123)), #Balanceo
        ('scaler', MinMaxScaler()) # Escalamiento
    ])

    # Modelos

    # Decision Tree
    #dt_pipeline = agregar_modelo(pipeline, DecisionTreeClassifier(random_state=123))
    #modelo = decisiontreeGS(dt_pipeline, X_train, y_train)
    #get_score(modelo, X_test, y_test, plot=True)
    #decisionTreePlot(X_train, y_train, X_test, y_test)
    
    # Ada Boost
    #ab_pipeline = agregar_modelo(pipeline, AdaBoostClassifier(algorithm="SAMME", random_state=123))
    #modelo = adaboostGS(ab_pipeline, X_train, y_train)
    #get_score(modelo, X_test, y_test, plot=True)
    #nested_cv(ab_pipeline, adaboostGS, X, y)
    #adaboostPlot(X_train, y_train, X_test, y_test)

    # Random forest
    #rf_pipeline = agregar_modelo(pipeline, RandomForestClassifier(bootstrap=True, random_state=123))
    #modelo = randomforestSinAfinamiento(rf_pipeline, X_train, y_train) # TRanfom forest sin afinacion de hyperparametros
    #modelo = randomforestRGS(rf_pipeline, X_train, y_train)
    #get_score(modelo, X_test, y_test, plot=False)
    #plotRandomForest(X_train, y_train, X_test, y_test)

main()
