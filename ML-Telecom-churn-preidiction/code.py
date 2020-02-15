# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 

# Code starts here
# Load the dataset
df = pd.read_csv(path)

# Split the dataset into features and target variable
X = df.drop(['customerID','Churn'], 1)

y = df['Churn'].copy()

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 0)




# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Code starts here

# Replace spaces with Nan
X_train['TotalCharges'].replace(' ', np.nan, inplace=True)

X_test['TotalCharges'].replace(' ', np.nan, inplace=True)

# Convert object to float
X_train['TotalCharges'] = X_train['TotalCharges'].astype(float)

X_test['TotalCharges'] = X_test['TotalCharges'].astype(float)

# Fill missing values with mean
X_train['TotalCharges'].fillna(X_train['TotalCharges'].mean(), inplace = True)

X_test['TotalCharges'].fillna(X_test['TotalCharges'].mean(), inplace = True)

# Check Nan values
print("Total NA values in training set :", X_train['TotalCharges'].isnull().sum())

print("Total NA values in test set :", X_test['TotalCharges'].isnull().sum())

# Encoding categorical columns
cat = X_train.select_dtypes(include='object').columns.tolist()

for i in cat:
    # Initialize label encoder
    enc = LabelEncoder()

    X_train[i] = enc.fit_transform(X_train[i])

for j in cat:
    # Intialize Label encoder
    enc = LabelEncoder()

    X_test[j] = enc.fit_transform(X_test[j])

# Replace yes and no with binary values

y_train.replace({'No': 0, 'Yes': 1}, inplace = True)

y_test.replace({'No': 0, 'Yes': 1}, inplace = True)



# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here

print(X_train.head())
print('-'*50)

print(X_test.head())
print('-'*50)

print(y_train)
print('-'*50)

print(y_test)
print('-'*50)

# Initialize an AdaBoost model
ada_model = AdaBoostClassifier(random_state = 0)

# Fit the model
ada_model.fit(X_train, y_train)

# Predictions
y_pred = ada_model.predict(X_test)

# Accuracy of the model
ada_score = accuracy_score(y_test, y_pred)
print("Accuracy of the predictions :", round(ada_score, 2))
print('-'*50)

# Confusion matrix
ada_cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix :\n", ada_cm)
print('-'*50)

# Classification Report
ada_cr = classification_report(y_test, y_pred)
print("Classification Report :\n", ada_cr)



# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here

print("*********************Model2 (XGBoost Classifier)***********************")

# Initialize an XGBoost classifier
xgb_model = XGBClassifier(random_state = 0)

# Fit the model 2
xgb_model.fit(X_train, y_train)

# Prediction for model 2
y_pred = xgb_model.predict(X_test)

# Accuracy score for model 2
xgb_score = accuracy_score(y_test, y_pred)
print("Accuracy of the prediction :", round(xgb_score, 2))
print('-'*50)

# Confusion Matrix for model 2
xgb_cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix :\n", xgb_cm)
print('-'*50)

# Classification Report for model 2
xgb_cr = classification_report(y_test, y_pred)
print("Classification Report :\n", xgb_cr)
print('-'*50)

print("*************Model3 (XGBoost Classifier using grid search)*************")

# Initialize Grid search model
xgb_clf = XGBClassifier(random_state = 0)
clf_model = GridSearchCV(estimator = xgb_clf, param_grid=parameters)

# Fit the grid model
clf_model.fit(X_train, y_train)

# Prediction for the grid model
y_pred = clf_model.predict(X_test)

# Accuracy of the grid model
clf_score = accuracy_score(y_test, y_pred)
print("Accuracy score of model 3 :", round(clf_score, 2))
print('-'*50)

# Confusion Matrix for the grid model
clf_cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix :\n", clf_cm)
print('-'*50)

# Classification Report for the grid model
clf_cr = classification_report(y_test, y_pred)
print("Classification Report :\n", clf_cr)



