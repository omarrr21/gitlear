
#SVR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data/Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)


from sklearn.svm import SVR
regression = SVR(kernel="rbf")
regression.fit(x,y)
y_pred = regression.predict(sc_x.transform([[6.5]]))
y_pred = sc_y.inverse_transform(y_pred)
print(y_pred)
# X_grid = np.arange(min(x),max(x),0.1)
# X_grid = X_grid.reshape(len(X_grid),1)

plt.scatter(x, y, color = 'red')
plt.plot(x, regression.predict(x), color = 'blue')
plt.title('Modelo de reg svr')
plt.xlabel('posicion del empleado')
plt.ylabel('Sueldo')
plt.show()