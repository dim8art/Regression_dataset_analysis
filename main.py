import math

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
def err(y, y_predicted):
    return 1-np.sum(np.power(np.subtract(y, y_predicted), 2))/np.sum(np.power(np.subtract(y, np.average(y)), 2))
def poly_predict(x, y, degree):
    fit = np.polyfit(x, y, deg=degree)
    Y_poly = np.polyval(fit, x)
    print(fit)
    return Y_poly
def exp_predict(x, y):
    fit = np.polyfit(x, np.log(y), 1)
    Y_exp = np.power(np.exp(fit[0]), x) * np.exp(fit[1])
    print(np.exp(fit[0]), np.exp(fit[1]))
    return Y_exp
a = np.array(pd.read_excel("gdp.xls"))
COUNTRY_INDEX = 0
x = np.arange(1960, 2021, 1)
y = np.array(a[COUNTRY_INDEX][4:65], float)
plt.scatter(x, y, c='blue')

plt.plot(x, poly_predict(x, y, 1), c='red')
plt.plot(x, poly_predict(x, y, 2), c='green')
plt.plot(x, poly_predict(x, y, 3), c='pink')
plt.plot(x, poly_predict(x, y, 4), c='brown')
plt.plot(x, poly_predict(x, y, 5), c='black')
plt.plot(x, exp_predict(x, y), c='magenta')
print(err(y,  poly_predict(x, y, 1)))
print(err(y,  poly_predict(x, y, 2)))
print(err(y,  poly_predict(x, y, 3)))
print(err(y,  poly_predict(x, y, 4)))
print(err(y,  poly_predict(x, y, 5)))
print(err(y,  exp_predict(x, y)))
plt.legend(["Test data ",
            "1 degree $R^2 =$" +str(err(y,  poly_predict(x, y, 1))),
            "2 degree $R^2 =$" +str(err(y,  poly_predict(x, y, 2))),
            "3 degree $R^2 =$" +str(err(y,  poly_predict(x, y, 3))),
            "4 degree $R^2 =$" +str(err(y,  poly_predict(x, y, 4))),
            "5 degree $R^2 =$" +str(err(y,  poly_predict(x, y, 5))),
            "Exp $R^2 =$" +str(err(y,  exp_predict(x, y)))], fontsize="x-small")
plt.show()

