import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
# Importing the dataset
dataset=pd.read_csv('Salary_Data.csv')

import seaborn as sns
#sns.heatmap(dataset.corr())
sns.heatmap(dataset.corr(),annot=True)
plt.show()
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values 


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, 
                     test_size = .25, random_state = 0) 


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred=regressor.predict(X_test)

from sklearn import metrics
print("Error=",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

print("Salary :",regressor.predict([[2.3]]))
##############

print("Intercept=",regressor.intercept_)
print("Coefficient=",regressor.coef_)
# print the R-squared value for the model


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


