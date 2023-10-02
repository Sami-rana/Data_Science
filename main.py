# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# plt.rcParams['figure.figsize'] = (20.0, 10.0)
# data = pd.read_csv('Salary_Data_Based_country_and_race.csv')
#
# print(data.shape)
# print(data.head())
#
# X = data['Salary'].values
# Y = data['Years of Experience'].values
#
# mean_x = np.mean(X)
# mean_y = np.mean(Y)
# m = len(X)
#
# numer = 0
# denom = 0
#
# for i in range(m):
#     numer += (X[i] - mean_x) * (Y[i] - mean_y)
#     denom += (X[i] - mean_x) ** 2
#
# b1 = numer / denom
# b0 = mean_y - (b1 * mean_x)
#
# print(b1, b0)
# import numpy as np
#
# X = [1, 2, 3, 'abc', 5]
# Y = [0.1, 0.2, 'def', 0.4, 0.5]
#
# # Clean the data by removing or replacing non-numerical values
# X_cleaned = [float(x) for x in X if isinstance(x, (int, float))]
# Y_cleaned = [float(y) for y in Y if isinstance(y, (int, float))]
#
# # Calculate the mean using the cleaned data
# mean_x = np.mean(X_cleaned)
# mean_y = np.mean(Y_cleaned)
#
# print("Mean of X:", mean_x)
# print("Mean of Y:", mean_y)
#
# max_x = np.max(X) + 100
# min_X = np.min(X) - 100
#
# x = np.linspace(min_X, max_x, 1000)
# y = b0 + b1 * x
#
# plt.plot(x, y, color='#58b970', label='Regression Line')
# plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')
# plt.xlabel('Salary Average')
# plt.ylabel('Years of Experience')
# print(plt.legend())
# print(plt.show())
#
# ss_t = 0
# ss_r = 0
# for i in range(m):
#     y_pred = b0 + b1 * X[i]
#     ss_t += (Y[i] - mean_y) ** 2
#     ss_r += (Y[i] - y_pred) ** 2
#
# r2 = 1 - (ss_r / ss_t)
#
# print(r2)

for item in zip([1,2,3,4,5], ['a','b','c','d','e']):
    print(item)
