# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here


# read the dataset
dataset = pd.read_csv(path)


# look at the first five columns
print(dataset.head())

# Check if there's any column which is not useful and remove it like the column id
dataset.drop('Id', axis=1, inplace=True)

# check the statistical description
dataset.describe()


# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 
cols = dataset.columns

#number of attributes (exclude target)
size = dataset.iloc[:,:-1].shape[1]

#x-axis has target attribute to distinguish between classes
x = dataset["Cover_Type"].copy()

#y-axis shows values of an attribute
y = dataset.iloc[:,:-1]

#Plot violin for all attributes
for i in range(0, size, 2):
    if len(cols) > i:
        plt.figure(figsize=[30,16])
        sns.violinplot(y[cols[i]], x)
        plt.title('Plotting the density of '+cols[i])
        plt.xticks(rotation=45)
        plt.xlabel(cols[i])
        plt.show()


# --------------
import numpy
upper_threshold = 0.5
lower_threshold = -0.5


# Code Starts Here
# Subset of continuous features
subset_train = dataset.iloc[:,:10]
print("Shape of the subset :", subset_train.shape)
print('-'*50)

# Correlation between features
data_corr = subset_train.corr()

# Visualising the Feature correlation
plt.figure(figsize = [13,9])
sns.heatmap(data_corr, annot=True)
plt.show()

# Storing the correlation values into Series
correlation = data_corr.unstack().sort_values(kind='quicksort')

# Iterating through the correlation values
corr_var_list = []

for j in correlation:
    if j > upper_threshold and j != 1:
        corr_var_list.append(j)
    elif j < lower_threshold and j != 1:
        corr_var_list.append(j)

# Display the features with high correlation
print("Correlation Values :\n", corr_var_list)
# Code ends here




# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
import numpy as np

# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)

# Split the data into features and target variables
X = dataset.drop('Cover_Type', axis=1)
Y = dataset.iloc[:,-1]

# Split into train and test set
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
    X, Y, test_size=0.2, random_state=0)

# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.


#Standardized
#Apply transform only for continuous data
# Initialize scaling
scaler = StandardScaler()
column = X_train.iloc[:,:10].columns

X_train_temp = X_train
X_test_temp = X_test

# Scaling training set
X_train_temp[column] = scaler.fit_transform(X_train_temp[column])
                                            
# Scaling test set
X_test_temp[column] = scaler.transform(X_test_temp[column])
                                            
#Concatenate scaled continuous data and categorical
X_train1 = X_train_temp

X_test1 = X_test_temp

scaled_features_train_df = X_train1

scaled_features_test_df = X_test1


# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


# Write your solution here:
# Initialize Select Percentile
skb = SelectPercentile(score_func=f_classif, percentile=90)

# Fit and transform
predictors = skb.fit_transform(X_train1, Y_train)

# Scores of the features
scores = skb.scores_

# Features
Features = numpy.asarray(X_train.columns.tolist())

dataframe = pd.DataFrame({'Features': Features, 'scores': scores}).sort_values('scores',ascending=False)

# Top predictors
top_p = dataframe['scores'].quantile(0.1)

top_k_predictors = dataframe[dataframe['scores'] > top_p]['Features'].tolist()

# Display results
print("Top Predictors :\n", top_k_predictors)


# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

# Initialize OneVsRestClassifier
clf1 = OneVsRestClassifier(LogisticRegression())
clf = OneVsRestClassifier(LogisticRegression())

# Model 1

print("**************** Model 1 (Using all the features) ****************")

# Fit the model on all features
model_fit_all_features = clf1.fit(X_train, Y_train)

# Predictions using all the features
predictions_all_features = model_fit_all_features.predict(X_test)

# Accuracy of the model
score_all_features = model_fit_all_features.score(X_test, Y_test)
print("Accuracy of the model with all the features :", score_all_features)
print('-'*50)

# Classification Report
clas1 = classification_report(Y_test, predictions_all_features)
print("Classification Report :\n", clas1)
print('-'*50)

# Confusion matrix
c_mat1 = confusion_matrix(Y_test, predictions_all_features)
print("Confusion matrix :\n", c_mat1)
print('-'*50)

# Precision
pre_score1 = precision_score(
    Y_test, predictions_all_features, average = 'weighted')
print("Precision of the model :", pre_score1)
print('-'*25, '- X -', '-'*25)

# Model 2

print("****************** Model 2 (Using Top scaled features) *****************")

# Fit the model on scaled features
model_fit_top_features = clf.fit(scaled_features_train_df[top_k_predictors], Y_train)

# Predictions
predictions_top_features = model_fit_top_features.predict(scaled_features_test_df[top_k_predictors])

# Accuracy of the model
score_top_features = model_fit_top_features.score(scaled_features_test_df[top_k_predictors], Y_test)
print("Accuracy of the model with top features(scaled) :", score_top_features)
print('-'*50)

# Classification Report
clas = classification_report(Y_test, predictions_top_features)
print("Classification Report :\n", clas)
print('-'*50)

# Confusion Matrix
c_mat = confusion_matrix(Y_test, predictions_top_features)
print("Confusion Matrix :\n", c_mat)
print('-'*50)

# Precision
pre_score = precision_score(
    Y_test, predictions_top_features, average = 'weighted')
print("Precision of the model :", pre_score)
print('-'*25, '- X -', '-'*25)


