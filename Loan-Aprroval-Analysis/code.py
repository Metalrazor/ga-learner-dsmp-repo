# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 



# code starts here
bank = pd.read_csv(path,sep=',',delimiter=None,header='infer')
categorical_var = bank.select_dtypes(include = 'object')
print(categorical_var)
numerical_var = bank.select_dtypes(include = 'number')
print(numerical_var)

# code ends here


# --------------
# code starts here
banks = bank.drop(['Loan_ID'],axis=1)
df = banks.isnull().sum()
print(df)
bank_mode = banks.mode()
for column in banks.columns:
    banks[column].fillna(banks[column].mode()[0], inplace=True)
print(banks)
#code ends herea


# --------------
# Code starts here
avg_loan_amount = pd.pivot_table(banks,index=['Gender','Married','Self_Employed'],values='LoanAmount',aggfunc='mean')
print(avg_loan_amount)


# code ends here



# --------------
# code starts here
loan_approved_se = banks[(banks['Self_Employed'] == 'Yes') & (banks['Loan_Status'] == 'Y')].count().iloc[0]
percentage_se = ((loan_approved_se * 100) / 614)
print(percentage_se)

loan_approved_nse = banks[(banks['Self_Employed'] == 'No') & (banks['Loan_Status'] == 'Y')].count().iloc[0]
percentage_nse = ((loan_approved_nse * 100) / 614)
print(percentage_nse)
# code ends here


# --------------
# code starts here
loan_term = banks['Loan_Amount_Term'].apply(lambda x : x/12)
print(loan_term)
big_loan_term = loan_term[loan_term >= 25].count()
print(big_loan_term)

# code ends here


# --------------
# code starts here
loan_groupby = banks.groupby('Loan_Status')

loan_groupby = loan_groupby['ApplicantIncome','Credit_History']

mean_values = loan_groupby.mean()
# code ends here


