#importing the libraries

import numpy as np
import pandas as pd
import matplotlib as plt

#reading the dataset

dataset = pd.read_csv('50_Startups.csv')

#putting indepandent variabls value

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4 ].values

#taking care of missing datasets

#from sklearn.impute import SimpleImputer

#imputer = SimpleImputer(missing_values=np.nan, strategy= 'mean')

#imputer.fit(X[:, 1:3])

##print(X)

#Encoding cetagoricl data

#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import LabelEncoder#, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3]=labelencoder_X.fit_transform(X[:, 3])
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#Avoiding dummy variable trap

X =X [:, 1:]

#labelencoder_Y = LabelEncoder()
#Y=labelencoder_Y.fit_transform(Y)
#print(Y)

#Spliting the data in testig and training

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#feature scalling

#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)

#Fitting data to multiple linear regression

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#predicting the test set result

Y_pred = regressor.predict(X_test)

#building the optimal backward elimination

#import statsmodels.formula.api as sm

import statsmodels.api as sm

X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis =1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]

#regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = X[:, [0, 1, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()













































