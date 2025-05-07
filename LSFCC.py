# Linear regression using the least squares method
import numpy as np
import pandas as pd

dataframe1 = pd.read_excel('LSF&CCSheets.xlsx')

x_values_process = []

y_values_process = []

weights = []

for i in range(dataframe1.shape[0]):
    x_values_process.append(dataframe1.loc[i, 'x values'])
    y_values_process.append(dataframe1.loc[i, 'y values'])
    weights.append(dataframe1.loc[i, 'weights'])

# x_values_input = input("Enter a list of x values (Enter with spaces, not commas):").split()

# y_values_input = input("Enter a list of y values (Enter with spaces, not commas):").split()

weight_or_not = input("Do you want to weight the points? (y/n):")

# for x in x_values_input:
#     x_values_process.append(float(x))

# for y in y_values_input:
#     y_values_process.append(float(y))

if weight_or_not == "n":
    weights = np.ones(len(x_values_process))
else:
    weights = weights

sum_wx = sum(w * x for w,x in zip(weights,x_values_process))

print("The sum of weighted x values = {:.4f}".format(sum_wx))

sum_wy = sum(w*y for w,y in zip(weights,y_values_process))

print("The sum of weighted y values = {:.4f}".format(sum_wy))

sum_wx_squared = sum(w * x**2 for w,x in zip(weights,x_values_process))

print("The sum of weighted x^2 values = {:.4f}".format(sum_wx_squared))

sum_wxy = sum(w*x*y for w,x, y in zip(weights,x_values_process, y_values_process))

print("The sum of weighted xy values = {:.4f}".format(sum_wxy))

sum_W = sum(weights)

print("The total weight/number of entries (N) = {}".format(sum_W))

delta = sum_W * sum_wx_squared - sum_wx**2

print("The value of weighted delta is = {}".format(delta))

A = (sum_wx_squared * sum_wy - sum_wx * sum_wxy) / (delta)
B = (sum_W * sum_wxy - sum_wx * sum_wy) / (delta)

dy = np.sqrt(1/(sum_W-2) * sum((y - A - B*x)**2 for x, y in zip(x_values_process, y_values_process)))

dA = dy * np.sqrt(sum_wx_squared / delta)

dB = dy * np.sqrt(sum_W/delta)

print("The best fit line is: y = {:.4f} + {:.4f}x".format(A, B))

print("The errors in the coefficients are: dA = {:.4f}, dB = {:.4f}".format(dA, dB))

print("The error in the y value is: dy = {:.4f}".format(dy))

x_dev = [x - np.mean(x_values_process) for x in x_values_process]
y_dev = [y - np.mean(y_values_process) for y in y_values_process]

covar = 1/(len(x_dev)) * (sum(x*y for x,y in zip(x_dev,y_dev)))

print("The covariance is = {:.4f}".format(covar))

print("The correlation coefficient is = {:.4f}".format(covar/(np.std(x_values_process)*np.std(y_values_process))))