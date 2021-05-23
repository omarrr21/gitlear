#regresion lineal multiple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('data/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],
    remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=float)
#Evitar la trampa de las variables ficticias
X = X[:,1:]

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#ajustar el modelo de regresion lineal multiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)
#prediccion de los resultados en el conjunto de testing
y_pred = regression.predict(X_test)

#construir el modelo optimo de RLM utilizando la eliminacion hacia atras
import statsmodels.api as sm
X = np.append(arr= np.ones((50,1)).astype(int),values=X, axis=1)

X_opt = X[:, [0,1,2,3,4,5]].tolist()
SL = 0.05
regression_OLS = sm.OLS(y,X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0,1,3,4,5]].tolist()
regression_OLS = sm.OLS(y,X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0,3]].tolist()
regression_OLS = sm.OLS(y,X_opt).fit()
regression_OLS.summary()