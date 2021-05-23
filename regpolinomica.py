#regresion polinomica
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data/Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#ajustar la regresion lineal con el dataset
from sklearn.linear_model import LinearRegression
Lin_reg = LinearRegression()
Lin_reg.fit(x, y)
#ajustar la regresion polinomica con el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(x)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#visualizacion de los resultados del modelo lineal
plt.scatter(x, y, color = 'red')
plt.plot(x, Lin_reg.predict(x), color = 'blue')
plt.title('Modelo de reg lineal')
plt.xlabel('posicion del empleado')
plt.ylabel('Sueldo')
plt.show()
#visualizacion de los resultados del modelo polinomico
X_grid = np.arange(min(x),max(x),0.1)
X_grid = X_grid.reshape(len(X_grid),1)

plt.scatter(x, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Modelo de reg polinomica')
plt.xlabel('posicion del empleado')
plt.ylabel('Sueldo')
plt.show()
#predicion de nuestros modelos

Lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))