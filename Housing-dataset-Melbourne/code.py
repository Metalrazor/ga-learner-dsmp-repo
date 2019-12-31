# --------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# path- variable storing file path
df = pd.read_csv(path)
df.head()

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 6)

corr = X_train.corr()
print("Correlation between all the independent features :",corr)
#Code starts here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Code starts here
# Initiating Regression model
regressor = LinearRegression()

# Fitting the line
regressor.fit(X_train, y_train)

# Prediction
y_pred = regressor.predict(X_test)

# R-squared score
r2 = r2_score(y_test, y_pred)
print("R-squared score of the model:", r2)



# --------------
from sklearn.linear_model import Lasso

# Code starts here
# Initiating Lasso model
lasso = Lasso()

# Fitting the lasso model
lasso.fit(X_train, y_train)

# Predicting the target variable using lasso model
lasso_pred = lasso.predict(X_test)

# R-squared score for the lasso model
r2_lasso = r2_score(y_test, lasso_pred)
print("R-squared score for the lasso model :", r2_score)


# --------------
from sklearn.linear_model import Ridge

# Code starts here
# Initiating a Ridge model
ridge = Ridge()

# Fitting the line to the Ridge model
ridge.fit(X_train, y_train)

# Ridge prediction
ridge_pred = ridge.predict(X_test)

# R-squared score for the Ridge model
r2_ridge = r2_score(y_test, ridge_pred)
print("R-squared score for the Ridge model :", r2_ridge)

# Code ends here


# --------------
from sklearn.model_selection import cross_val_score

#Code starts here
# Initiating a Linear Regression model for cross validation
regressor = LinearRegression()

# Cross validation
score = cross_val_score(regressor, X_train, y_train, cv = 10)

# Average RMSE in the model
mean_score = np.mean(score)
print("Average error in the model :", mean_score)


# --------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Code starts here
# Initiating a pipeline for polynomial features
model = make_pipeline(PolynomialFeatures(2), LinearRegression())

# Fitting the model
model.fit(X_train, y_train)

# Prediction for the model
y_pred = model.predict(X_test)

# R-squared score for the model
r2_poly = r2_score(y_test, y_pred)
print("R-squared score for the model :", r2_poly)


