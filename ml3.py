
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
from sklearn.model_selection import cross_val_score,GridSearchCV
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

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# Configuracion de pandas para imprimir todo el data set
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_colwidth', None)

# Cargar el csv
def load():
  df = pd.read_csv("yeast.data", sep='\s+', header=None)
  columns = ["Sequence_Name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "localization_site"]
  df.columns = columns
  return df

# Analisis exploratorio

# Graficar una variables
def plotTarget(data, target):
  categoria_counts = data[target].value_counts()
  porcentajes = data[target].value_counts(normalize=True) * 100
  print("Cantidad: ", categoria_counts)
  print("Porcentajes: ", porcentajes)
  categorias_ordenadas = categoria_counts.index
  plt.figure(figsize=(8, 6))
  plt.bar(categorias_ordenadas, categoria_counts, color='skyblue')
  plt.xlabel('Categoría')
  plt.ylabel('Frecuencia')
  plt.title('Distribución de Categorías')
  plt.show()

# Grafico de barras variables categoricas
def analisisCategoricas(df):
  varCategoricas = df.select_dtypes(exclude=np.number).columns
  print("Categoricas: ", varCategoricas)
  n_vars = len(varCategoricas)
  n_cols = 3
  n_rows = (n_vars + n_cols - 1) // n_cols  if  (n_vars + n_cols - 1) // n_cols > 0 else 1
  fig, axis = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
  index = 0
  for i in range(n_rows):
    for j in range(n_cols):
      if index < n_vars:
        ax = sns.countplot(x=varCategoricas[index], data=df, ax=axis[i][j])
        if varCategoricas[index] in ['job']:
          for item in ax.get_xticklabels():
            item.set_rotation(15)
        for p in ax.patches:
          height = p.get_height()
          ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/len(df)*100),
            ha="center")
        index += 1
      else:
        axis[i, j].set_visible(False)
  plt.tight_layout()
  plt.show()

def plotCategorica(datos):
  valores_unicos, recuentos = np.unique(datos, return_counts=True)

  # Crear el gráfico de barras
  plt.bar(valores_unicos, recuentos)

  # Agregar etiquetas y título
  plt.xlabel('Valor')
  plt.ylabel('Recuento')
  plt.title('Recuento de valores')

  # Mostrar el gráfico
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

# Analisis de nulos
def nullAnalysis(df):
  null_columns=df.isnull().any()
  print("Nulos en columnas:")
  print(null_columns)
  null_sum = df.isnull().sum()
  print("Suma de nulos:")
  print(null_sum)

# Escalamiento

#  Escalamiento con StandardScaler
def standardScaler(df, target):
  y = df[target]
  X = df.drop(columns=[target])
  scaler = StandardScaler()
  df_scaled = scaler.fit_transform(X)
  df_scaled = pd.DataFrame(df_scaled, columns=X.columns, index=df.index)
  df_final = df_scaled.join(y)
  return df_final

#  Escalamiento con MinMaxScaler
def minMaxScaler(df, target):
  y = df[target]
  X = df.drop(columns=[target])
  scaler = MinMaxScaler()
  df_scaled = scaler.fit_transform(X)
  df_scaled = pd.DataFrame(df_scaled, columns=X.columns, index=df.index)
  df_final = df_scaled.join(y)
  return df_final
  
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

# Logistic Regression

#  Logistic regression grid search
def decisionTreeTunning(data, target):

  train_val,test=train_test_split(data, test_size=0.1,random_state=42)
  train,validate=train_test_split(train_val, test_size=0.2,random_state=42)
  
  XTrain=train.drop(target,axis="columns")
  yTrain=train[target]
  XTest=test.drop(target,axis="columns")
  yTest=test[target]
  XVal=validate.drop(target,axis="columns")
  yVal=validate[target]
  
  accEntropyTrain=[]
  accEntropy=[]
  accGiniTrain=[]
  accGini=[]
  accEntropy=[]
  maxDepth=[]
  
  accEntTrain=[]
  
  for i in range(1,40):
      #Entropy
      model=DecisionTreeClassifier(criterion="entropy",max_features="log2",max_depth=i,random_state=2)
      model.fit(XTrain,yTrain)
      # Train accuracy
      pred=model.predict(XTrain)
      acc=accuracy_score(yTrain,pred)
      accEntropyTrain.append(acc)
      # Validate accuracy
      pred=model.predict(XVal)
      acc=accuracy_score(yVal,pred)
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

  d=pd.DataFrame({"acc_entropyT":pd.Series(accEntTrain),"acc_entropy":pd.Series(accEntropy),"max_depth":pd.Series(maxDepth)})
    
  plt.plot("max_depth","acc_entropy",'b',data=d,label="entropy")
  plt.plot("max_depth","acc_entropyT",'r',data=d,label="entropy_train")
  plt.xlabel("max_depth")
  plt.ylabel("accuracy")
  plt.legend()

  '''
  dt=DecisionTreeClassifier(criterion="entropy",max_features="log2",max_depth=5,random_state=2)
  dt.fit(XTrain,yTrain)
  pred=model.predict(XTest)
  acc=accuracy_score(yTest,pred)
  print(acc)
  print(dt.tree_.threshold)
  '''

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
  dt = DecisionTreeClassifier(random_state=123)
  param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_features': [None, 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10]
  }
  grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2)
  grid_search.fit(X_train, y_train)
  print("Hiperparametros ",grid_search.best_params_)
  print("Ccore ",grid_search.best_score_)

  best_model = grid_search.best_estimator_
  y_pred = best_model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Precisión en el conjunto de prueba: {accuracy}")
  return grid_search.best_score_

#  Logistic regression cross-validation metrics
def decisionTreeCV(X, y):
    # Logistic regresion
    X = X.values
    dt=DecisionTreeClassifier(criterion= 'gini', max_features='log2', max_depth= 10, min_samples_leaf= 2, min_samples_split= 20, random_state=123)
    skf=KFold(n_splits=10,shuffle=True,random_state=123)

    # Calcular todas las etiquetas
    true_labels = []
    predicted_labels = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        true_labels.extend(y_test)
        predicted_labels.extend(y_pred)

    # Reporte de clasificación
    print("\nReporte de Clasificación:")
    scores = cross_val_score(dt, X, y, cv=skf, scoring='f1_weighted')
    f1_weighted = scores.mean()
    print(classification_report(true_labels, predicted_labels))
    print("F1-SCORE: ", f1_weighted)

    # Matriz de confusión
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    mapping = {'CYT': 0, 'NUC': 1, 'MIT': 2, 'ME3': 3, 'ME2': 4, 'ME1': 5, 'EXC': 6, 'VAC': 7, 'POX': 8, 'ERL': 9}
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False,
                xticklabels=mapping,
                yticklabels=mapping)
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Real')
    plt.title('Matriz de Confusión')
    plt.show()

# SVC

#  SVC grid search
def svcGS(X,y):
  svc=SVC(probability=True, random_state=123, class_weight='balanced')
  param_grid = [
      {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear']},
      {'C': [0.01, 0.1, 1, 10], 'kernel': ['poly'], 'gamma': [0.1, 0.01, 0.001, 'scale', 'auto'], 'degree': [2, 3, 4, 5], 'coef0': [0.0, 0.1, 0.5, 1.0]},
      {'C': [0.01, 0.1, 1, 10], 'kernel': ['rbf'], 'gamma': [0.1, 0.01, 0.001, 'scale', 'auto']},
      {'C': [0.01, 0.1, 1, 10], 'kernel': ['sigmoid'], 'gamma': [0.1, 0.01, 0.001, 'scale', 'auto'], 'coef0': [0.0, 0.1, 0.5, 1.0]}
  ]
  skf = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
  grid_search=GridSearchCV(estimator=svc, param_grid=param_grid,cv=skf, scoring='f1_weighted', refit = True, verbose=2)
  grid_search.fit(X,y)
  
  print("Hiperparametros ",grid_search.best_params_)
  print("score ",grid_search.best_score_)
  return grid_search.best_score_

#  SVC cross validation metrics
def svcCV(X,y):
    X = X.values
    #svc=SVC(C=1,kernel='sigmoid', gamma='auto', coef0=0.5, probability=True, random_state=123, class_weight='balanced')
    svc=SVC(C=0.1,kernel='poly', degree=2, gamma=0.1, coef0=1.0, probability=True, random_state=123, class_weight='balanced')
    skf=StratifiedKFold(n_splits=10,shuffle=True,random_state=1)

    # Calcular todas las etiquetas
    true_labels = []
    predicted_labels = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)
        true_labels.extend(y_test)
        predicted_labels.extend(y_pred)

    # Reporte de clasificación
    print("\nReporte de Clasificación:")
    scores = cross_val_score(svc, X, y, cv=skf, scoring='f1_weighted')
    f1_weighted = scores.mean()
    print(classification_report(true_labels, predicted_labels))
    print("F1-SCORE: ", f1_weighted)

    # Matriz de confusión
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    mapping = {'CYT': 0, 'NUC': 1, 'MIT': 2, 'ME3': 3, 'ME2': 4, 'ME1': 5, 'EXC': 6, 'VAC': 7, 'POX': 8, 'ERL': 9}
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False,
                xticklabels=mapping,
                yticklabels=mapping)
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Real')
    plt.title('Matriz de Confusión')
    plt.show()
  
# Feature Selection

  # Algoritmo forward selection
def forward_selection(X, y, threshold=0.01):
    X_int = pd.DataFrame({'intercept': np.ones(len(X))}).join(X)
    included = ['intercept']
    excluded = list(set(X_int.columns) - set(included))
    best_features = []
    current_score = 0.0
    while excluded:
        scores_with_candidates = []
        for feature in excluded:
            model_features = included + [feature]
            X_subset = X_int[model_features]
            model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
            scores = cross_val_score(model, X_subset, y, cv=5, scoring='accuracy')
            mean_score = np.mean(scores)
            scores_with_candidates.append((mean_score, feature))
        scores_with_candidates.sort(reverse=True)
        best_score, best_feature = scores_with_candidates.pop()
        if best_feature is None or best_score <= current_score + threshold:
            break
        included.append(best_feature)
        excluded.remove(best_feature)
        best_features.append((best_feature, best_score))
        current_score = best_score
    print(best_features)
    return included, best_features

  # Algoritmo Recursive Forward Elimination
def selectFeatures(X,y, n_features):
  model=LogisticRegression()
  rfe=RFE(model,n_features_to_select=n_features)
  fit=rfe.fit(X, y)
  print("selected features ",X.columns[fit.support_])


def main():
    
    df = load()
    
    # Dropear ID
    df = df.iloc[:, df.columns != 'Sequence_Name'] 
    
    # Aplicar feature selection
    #  Forward selection
    #df = df[['Status', 'Drug', 'Age', 'Sex', 'Platelets', 'Tryglicerides', 'Edema']] 
    #  Recursive Forward Elimination
    #df = df[['Status', 'Drug','N_Days', 'Age', 'Bilirubin', 'Alk_Phos', 'Platelets', 'Prothrombin', 'Stage', 'Sex', 'Ascites', 'Hepatomegaly']]
             
    #Analisis exploratorio
    #analisisNumericas(df)
    #plotTarget(df, 'localization_site')

    # Analisis de nulos
    nullAnalysis(df)

    # Encoding
    mapping = {
      'CYT': 0,
      'NUC': 1,
      'MIT': 2,
      'ME3': 3,
      'ME2': 4,
      'ME1': 5,
      'EXC': 6,
      'VAC': 7,
      'POX': 8,
      'ERL': 9,
    }
    df = encodingLabel(df, 'localization_site', mapping)
    
    # Escalamiento
    df=standardScaler(df, 'localization_site')
    #df=minMaxScaler(df, 'Status')
    
    # Tratamiento de outliers    
    #df = tratamientoOutliers(df, 'localization_site', contamination=0.07, plot=False)

    # Balanceo de datos
    #  xd

    '''
    # X,y para los modelos
    #X = df.drop('localization_site',axis=1)
    #y = df['localization_site']
    # Reduccion a clasificacion binaria
    #y=y.apply(lambda x:1 if x<2 else 0) # 0:vivo, 1:muerto
    '''
    # Decision tree
    lr = decisionTreeTunning(df, 'localization_site') 
    #decisionTreeCV(X,y)
    '''
    # Ada Boost
    lr = logisticGS(X,y) 
    logisticCV(X,y)

    # Random Forest
    svc = svcGS(X,y)
    svcCV(X, y)

    # Feature selection
    forward_selection(X, y, 0.001)
    selectFeatures(X, y, 10)
    '''

main()