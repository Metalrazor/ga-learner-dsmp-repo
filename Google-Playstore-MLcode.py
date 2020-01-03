# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
# Loading the dataset
data = pd.read_csv(path)

# Plotting a histogram for the Rating feature
data['Rating'].plot(kind='hist')
plt.show()

# Cleaning the outliers in Ratings feature
data = data[data['Rating'] <= 5]

# Plotting a histogram for the cleaned dataset
data['Rating'].plot(kind='hist')
plt.show()

#Code ends here


# --------------
# code starts here
# Checking for NaN values
total_null = data.isnull().sum()

# Percentage NaN values in each column
percent_null = (total_null / data.isnull().count())

# Creating a dataframe of the missing data
missing_data = pd.concat([total_null, percent_null], axis=1, keys=['Total','Percent'])

# Display the dataset
print("Initial Missing data :\n",missing_data)
print('-'*50)
# Dropping rows with NaN values
data_1 = data.dropna(axis=0)

# Checking for NaN values in the cleaned data
total_null_1 = data_1.isnull().sum()

# Percentage NaN values in each column
percent_null_1 = (total_null_1 / data_1.isnull().count())

# Creating a dataframe of the cleaned data
missing_data_1 = pd.concat([total_null_1, percent_null_1], axis=1, keys=['Total','Percent'])

# Display the cleaned dataset
print("Final Missing data :\n",missing_data_1)
# code ends here


# --------------

#Code starts here
sns.catplot(x="Category", y="Rating", data=data_1, kind="box", height=10)
plt.xticks(rotation=90)
plt.title("Rating vs. Category [BoxPlot]")
plt.show()
#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
data['Installs'].value_counts()

data['Installs'] = data['Installs'].str.replace(',',"")
data['Installs'] = data['Installs'].str.replace('+',"")
print(data['Installs'])

data['Installs'] = pd.to_numeric(data['Installs'], downcast='signed')
print(data['Installs'])

le = LabelEncoder()

data['Installs'] = le.fit_transform(data['Installs'])

sns.regplot(x="Installs", y="Rating", data=data)
plt.title('Rating vs. Installs [RegPlot]')
plt.show()
#Code ends here



# --------------
#Code starts here
# Counting all the unique values in Price column
data['Price'].value_counts()

# Removing the '$' sign from the Price column
data['Price'] = data['Price'].str.replace('$','')

# Converting the Price column to dataype float
data['Price'] = pd.to_numeric(data['Price'], downcast='float')

# Plotting a regression plot for Rating against Price
plt.figure(figsize=[15,9])
sns.regplot(x='Price', y='Rating', data=data)
plt.title('Rating vs. Price [Regplot]')
plt.show()
#Code ends here


# --------------

#Code starts here
# Unique values of the column Genre
data['Genres'].unique()

# Splittng the values by ';' in Genres column and keeping only the first genre
data['Genres'] = data['Genres'].str.split(';', n=1, expand=True)[0]

# Grouping the dataset by genres
gr_data = data.groupby('Genres', as_index=False)

# Grouping the Genres and rating features
gr_data = gr_data['Genres','Rating']

# Average rating for each genre
gr_mean = gr_data.mean()

# Display the statistics
gr_mean.describe()

# Sorting values by Rating feature
gr_mean = gr_mean.sort_values('Rating')

# Display the genre with the highest and lowest rating
print("The genre with the highest rating :\n", gr_mean.max())
print("The genre with the lowest rating :\n", gr_mean.min())
#Code ends here


# --------------

#Code starts here
# Visualising the data in Last Updated column
data['Last Updated']

# Conversion to datetime format
data['Last Updated'] = pd.to_datetime(data['Last Updated'])

# Recent update
max_date = data['Last Updated'].max()

# Variable representing the difference in days
Diff_dates = max_date - data['Last Updated']

# Creating a new feature representing the difference in the days of update release from the recently updated app
data['Last Updated Days'] = Diff_dates.dt.days
print(data['Last Updated Days'])

# Plotting a Regression plot for rating against Last Updated days
plt.figure(figsize=[20,13])
sns.regplot(x='Last Updated Days', y='Rating', data=data)
plt.title('Rating vs. Last Updated Days[RegPlot]')
plt.show()
#Code ends here


