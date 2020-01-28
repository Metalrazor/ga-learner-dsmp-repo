# --------------
#Importing header files

import pandas as pd
from sklearn.model_selection import train_test_split


# Code starts here
# Load the dataset
data = pd.read_csv(path)

# Subset the data
X = data.drop(['customer.id','paid.back.loan'], axis=1)
y = data['paid.back.loan'].copy()

# Split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Code ends here


# --------------
#Importing header files
import matplotlib.pyplot as plt

# Code starts here
# Count the categories in target variable
fully_paid = y_train.value_counts()

# Visualize the target variable
plt.figure(figsize = [14,10])
fully_paid.plot.bar()
plt.show()

# Code ends here


# --------------
#Importing header files
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Code starts here
# Convert the interest rate feature into float type
X_train['int.rate'] = X_train['int.rate'].str.replace('%','').astype(float)
X_test['int.rate'] = X_test['int.rate'].str.replace('%','').astype(float)

# Dividing the interest rate with 100
X_train['int.rate'] = X_train['int.rate']/100
X_test['int.rate'] = X_test['int.rate']/100

# Subset numerical features
num_df = X_train.select_dtypes(include='number')

# Subset categorical features
cat_df = X_train.select_dtypes(include=np.object)

# Code ends here


# --------------
#Importing header files
import seaborn as sns


# Code starts here
# List of all numerical features
cols = num_df.columns

# Plotting the boxplots for all the features against the target
fig, axes = plt.subplots(nrows = 9, ncols = 1, figsize=[8,45])

for i in range(0,9):
    sns.boxplot(x=y_train, y=num_df[cols[i]], ax=axes[i])

# Code ends here


# --------------
# Code starts here
# List of all the categorical features
cols = cat_df.columns

# Plotting boxplots for all the categorical features against the target
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=[15,10])

for i in range(0,2):
    for j in range(0,2):
        sns.countplot(x=X_train[cols[i*2+j]], hue=y_train, ax=axes[i,j])
# Code ends here


# --------------
#Importing header files
from sklearn.tree import DecisionTreeClassifier

# Code starts here
# Fill the null values with NA in training set(categorical features)
for i in cols:
    X_train[i].fillna('NA')
    
    # Initialize Label encoder
    le = LabelEncoder()
    
    # Fit and transform the values in training set(categorical features)
    X_train[i] = le.fit_transform(X_train[i])

# Fill the null values with NA in testing set(categorical features)
for i in cols:
    X_test[i].fillna('NA')
    
    # Initialize Label encoder
    le1 = LabelEncoder()
    
    # Transform the values in testing set(categorical features)
    X_test[i] = le1.fit_transform(X_test[i])

# Replace Yes and No with 0 and 1 in target variable
y_train = pd.Series(np.where(y_train.values == 'Yes', 1, 0), y_train.index)
y_test = pd.Series(np.where(y_test.values == 'Yes', 1, 0), y_test.index)

# Initialize a DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=0)

# Fit the model with training set
model.fit(X_train, y_train)

# Accuracy of the model
acc = model.score(X_test, y_test)
print("Accuracy of the model :", acc)

# Code ends here


# --------------
#Importing header files
from sklearn.model_selection import GridSearchCV

#Parameter grid
parameter_grid = {'max_depth': np.arange(3,10), 'min_samples_leaf': range(10,50,10)}

# Code starts here
# Initialize a DecisionTreeClassifier
model_2 = DecisionTreeClassifier(random_state=0)

# Initialize Cross validation object
p_tree = GridSearchCV(estimator=model_2, param_grid=parameter_grid, cv=5)

# Fit the grid model to training set
p_tree.fit(X_train, y_train)

# Accuracy of the grid model
acc_2 = p_tree.score(X_test, y_test)
print("Accuracy of the grid model :", acc_2)

# Code ends here


# --------------
#Importing header files

from io import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus

# Code starts here
# Initialize the parameters for decision tree graph
dot_data = export_graphviz(decision_tree=p_tree.best_estimator_, out_file=None,
 feature_names=X.columns, filled = True,
 class_names=['loan_paid_back_yes','loan_paid_back_no'])

# Draw Decision tree graph
graph_big = pydotplus.graph_from_dot_data(dot_data)

# show graph - do not delete/modify the code below this line
img_path = user_data_dir+'/file.png'
graph_big.write_png(img_path)

plt.figure(figsize=(20,15))
plt.imshow(plt.imread(img_path))
plt.axis('off')
plt.show() 

# Code ends here


