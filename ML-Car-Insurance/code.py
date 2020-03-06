# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
# Load the dataset
df = pd.read_csv(path)

df.head()

# Feature Information
df.info()

# Removing '$' and ',' from the features
cols = ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']

for i in cols:
    if i in df.columns:
        df[i] = df[i].str.replace('$','')
        df[i] = df[i].str.replace(',','')

# Splitting the data into features and target
X = df.drop('CLAIM_FLAG', axis=1)

y = df['CLAIM_FLAG'].copy()

# Count of the values in target
y.value_counts()

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 6)

# Code ends here


# --------------
# Code starts here
# Convert the following columns to float type
for i in cols:
    if i in X_train.columns:  ## Training set
        X_train[i] = np.asarray(X_train[i]).astype(np.float)

for i in cols:
    if i in X_test.columns:  ## Testing set
        X_test[i] = np.asarray(X_test[i]).astype(np.float)

# Check for null values
print("Null values in X_train :\n", X_train.isnull().sum())
print("-"*50)
print("Null values in X_test :\n", X_test.isnull().sum())
# Code ends here


# --------------
# Code starts here
# Dropping the Null values
X_train.dropna(subset=['YOJ','OCCUPATION'], inplace=True)

X_test.dropna(subset=['YOJ','OCCUPATION'], inplace=True)

# Reindexing the target
y_train = y_train[X_train.index]

y_test = y_test[X_test.index]

# Fill the Null values

cols2 = ['AGE','CAR_AGE','INCOME','HOME_VAL']

for i in cols2:
    if i in X_train:
        X_train[i].fillna(X_train[i].mean(), inplace=True)

for i in cols2:
    if i in X_test:
        X_test[i].fillna(X_test[i].mean(), inplace=True)
# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
# Encoding object types in train set
for i in columns:
    if i in X_train.columns:

        # Initiate Label Encoding object
        le = LabelEncoder()
        
        # Encode the data with object type data
        X_train[i] = le.fit_transform(X_train[i].astype(str))

# Encoding object types in Test set
for i in columns:
    if i in X_test.columns:

        # Initiate Label Encoding object
        le = LabelEncoder()
        
        # Encode the data with object type data
        X_test[i] = le.fit_transform(X_test[i].astype(str))

# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 

print("************************Logistic Regression************************")

# Initiate Logistic Regression Model
model = LogisticRegression()

# Fit the model
model.fit(X_train, y_train)

# Predictions for the model
y_pred = model.predict(X_test)

# Accuracy for the predictions
score = accuracy_score(y_test, y_pred)
print("Accuracy :", round(score, 2))
# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
# Initiate SMOTE object
smote = SMOTE(random_state=9)

# Fit the sample
X_train, y_train = smote.fit_sample(X_train, y_train)

# Initiate a Standard Scaler
scaler = StandardScaler()

# Scale the features
X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)

# Code ends here


# --------------
# Code Starts here

print("***********************Logistic Regression 2***********************")

# Initiate Logistic Regression model
model = LogisticRegression()

# Fit the model on training data
model.fit(X_train, y_train)

# Predictions for the model
y_pred = model.predict(X_test)

# Accuracy score for the model
score = accuracy_score(y_test, y_pred)
print("Accuracy :", round(score, 2))

# Code ends here


