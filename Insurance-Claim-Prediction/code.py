# --------------
# import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Code starts here
# Load the dataset
df = pd.read_csv(path)
df.head()

# Split the dataset into features and target variables
X = df.iloc[:,:7]
y = df.iloc[:,7]

# Further split it into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 6 )

# Code ends here


# --------------
import matplotlib.pyplot as plt


# Code starts here
# Visualize the distribution of the bmi for all the claimants
X_train['bmi'].plot(kind='box')
plt.show()
print('-'*50)

# Set Quantile to 0.95
q_value = X_train['bmi'].quantile(q=0.95)
print("95th percentile value :",round(q_value, 2))
print('-'*50)

# Count of unique values in the target feature
print("Count of unique Target values :\n", y_train.value_counts())
# Code ends here


# --------------
# Code starts here
# Correlation between features
relation = X_train.corr()
print("Correlation table :\n", relation)
print('-'*50)

# Visualize the correlation between the features
sns.pairplot(X_train)
plt.show()
# Code ends here


# --------------
import seaborn as sns
import matplotlib.pyplot as plt

# Code starts here
# List of features to be plotted
cols = ['children', 'sex', 'region', 'smoker']

# Creating subplots
fig, axes = plt.subplots(nrows = 2, ncols = 2)

# Iterating through rows and columns
for i in range(0,2):
    for j in range(0,2):
        col = cols[i * 2 + j]
        sns.countplot(x=X_train[col], hue = y_train, ax = axes[i,j])

# Code ends here


# --------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# parameters for grid search
parameters = {'C':[0.1,0.5,1,5]}

# Code starts here
# Initiate a Logistic Regression model
lr = LogisticRegression(random_state = 9)

# Grid search with the parameters given
grid = GridSearchCV(lr, param_grid = parameters)

# Fit the model to the training set
grid.fit(X_train, y_train)

# Predict values
y_pred = grid.predict(X_test)

# Accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy :", accuracy)

# Code ends here


# --------------
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# Code starts here
# ROC AUC score of the model
score = roc_auc_score(y_test, y_pred)
print("ROC AUC score of the model :", score)
print('-'*50)

# Predict the Probability of the predicted variable
y_pred_p = pd.DataFrame(grid.predict_proba(X_test), columns=['A','B'])
y_pred_proba = y_pred_p.iloc[:,1]

# FPR and TPR values
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
print("FPR :", fpr)
print('-'*50)
print("TPR :", tpr)
print('-'*50)

# ROC AUC score for the Predicted probabilities
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC AUC score for the predicted probabilities :", roc_auc)
print('-'*50)

# Visualize the ROC AUC curve
plt.figure(figsize=[15,10])
plt.plot(fpr, tpr, label="Logistic model, auc="+str(roc_auc))
plt.show()
# Code ends here


