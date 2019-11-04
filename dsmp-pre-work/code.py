# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'

#New record
new_record= [[50,  9,  4,  1,  0,  0, 40,  0]]

#Code starts here
data = np.genfromtxt(path, dtype = 'int64', delimiter=",", skip_header=1)
print("\nData: \n\n", data)
print("\nType of data: \n\n", type(data))

new_record_1 = np.asarray(new_record)
print(type(new_record_1))
print(new_record_1.dtype)

census = np.concatenate((data,new_record_1), axis = 0)
print(census)


# --------------
#Code starts here
age = census[:,0]
print(age)

# Maximum age in the age group
max_age = age.max()
print(max_age)

# Minimum age in the age group
min_age = age.min()
print(min_age)

# Average age
age_mean = age.mean()
print(age_mean)

# Standard Deviation of Age group
age_std = age.std()
print(age_std)


# --------------
#Code starts here
# Various races in the country
race = np.array(census[:,2],dtype = np.int64)

race_0 = census[census[:,2] == 0]
race_1 = census[census[:,2] == 1]
race_2 = census[census[:,2] == 2]
race_3 = census[census[:,2] == 3]
race_4 = census[census[:,2] == 4]

print(race_0, race_1, race_2, race_3, race_4)

# Summation of races
len_0, len_1, len_2, len_3, len_4 = len(race_0), len(race_1), len(race_2), len(race_3), len(race_4)
print(len_0, len_1, len_2, len_3, len_4)

# Minority race
minority_races = np.asarray([len_0, len_1, len_2, len_3, len_4], dtype = np.int64)

minority = minority_races == min(minority_races)

minority_race = minority_races.tolist().index(minority_races[minority])

print(minority_race)




# --------------
#Code starts here

senior_citizens = census[census[:,0] > 60]

working_hours_sum = sum(senior_citizens[:,6])

senior_citizens_len = len(senior_citizens)

avg_working_hours = working_hours_sum / senior_citizens_len

print(avg_working_hours)


# --------------
#Code starts here

# Education and qualification
high = census[census[:,1] > 10] 
low = census[census[:,1] <= 10]

# Average salaries earned by the citizens
avg_pay_high = high[:,7].mean()
avg_pay_low = low[:,7].mean()

# Required Education
avg_pay_high > avg_pay_low


