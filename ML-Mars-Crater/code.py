# --------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Code starts here

# Load the dataset
df = pd.read_csv(path)
df.head()

# Split into features and target
X = df.drop('attr1089', 1)
y = df['attr1089'].copy()

# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 4)

# Initialing scaler
scaler = MinMaxScaler()

# Fit the scaler on training features
scaler.fit(X_train)

# Transform the features
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

# Code ends here


# --------------
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Initialize Logistic Regression model
lr = LogisticRegression()

# Fit the model on training set
lr.fit(X_train, y_train)

# Predict the target for test features
y_pred = lr.predict(X_test)

# Metric check
roc_score = roc_auc_score(y_test, y_pred)

print("ROC_AUC_Curve :", round(roc_score, 2))



# --------------
from sklearn.tree import DecisionTreeClassifier

# Initialize Decision tree model
dt = DecisionTreeClassifier(random_state = 4)

# Fit the model
dt.fit(X_train, y_train)

# Predict the target
y_pred = dt.predict(X_test)

# Metric check
roc_score = roc_auc_score(y_test, y_pred)

print("Metric score of the decision tree model :", round(roc_score, 2))



# --------------
from sklearn.ensemble import RandomForestClassifier


# Code strats here

# Initiate a Random Forest model
rfc = RandomForestClassifier(random_state = 4)

# Fit the model
rfc.fit(X_train, y_train)

# Predict the target for test features
y_pred = rfc.predict(X_test)

# Metric check
roc_score = roc_auc_score(y_test, y_pred)

print("Metric score of the Random Forest model :", round(roc_score, 2))

# Code ends here


# --------------
# Import Bagging Classifier
from sklearn.ensemble import BaggingClassifier


# Code starts here
# Initialize Bagging model
bagging_clf = BaggingClassifier(
    base_estimator = DecisionTreeClassifier(), n_estimators = 100, 
    max_samples = 100, random_state = 0
)

# Fit the model on training set
bagging_clf.fit(X_train, y_train)

# Accuracy of the model
score_bagging = bagging_clf.score(X_test, y_test)

print("Accuracy of the model :", round(score_bagging, 2))
# Code ends here


# --------------
# Import libraries
from sklearn.ensemble import VotingClassifier

# Various models
clf_1 = LogisticRegression()
clf_2 = DecisionTreeClassifier(random_state=4)
clf_3 = RandomForestClassifier(random_state=4)

model_list = [('lr',clf_1),('DT',clf_2),('RF',clf_3)]


# Code starts here

# Initialize Ensemble model
voting_clf_hard = VotingClassifier(estimators = model_list, voting = 'hard')

# Fit the model
voting_clf_hard.fit(X_train, y_train)

# Accuracy of the model
hard_voting_score = voting_clf_hard.score(X_test, y_test)

print("Accuracy of the model by hard voting :", round(hard_voting_score, 2))

# Code ends here


