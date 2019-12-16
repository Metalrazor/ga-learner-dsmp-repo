# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df = pd.read_csv(path)

# Probability that the borrower's fico score above 700
p_a = len(df[df['fico'] > 700]) / len(df)

# Probability that the purpose of the borrower is to consolidate debt
p_b = len(df[df['purpose'] == 'debt_consolidation']) /len(df)

# Subset showing only borrowers with purpose for debt consolidation
df1 = df[df['purpose'] == 'debt_consolidation']

# Probability that borrowers with purpose as debt consolidation given that their fico score is above 700
p_a_b = len(df1[df1['fico'] > 700]) / len(df1)

# Check for independency
result = p_a_b == p_b
print("If the purpose of a borrower is debt consolidation then does his fico score get afftected? \n Answer :", result)
# code ends here


# --------------
# code starts here
# Probability that the loan has been paid back
S = df
A = df[df['paid.back.loan'] == 'Yes']
prob_lp = len(A) / len(S)
print("Probability that the loan has been paid back is",round(prob_lp, 2))
print('-'*80)

# Probability that the customer has met the credit underwriting criteria of LendingClub.com
B = df[df['credit.policy'] == 'Yes']
prob_cs = len(B) / len(S)
print("Probability that the customer has met the credit underwriting criteria is",round(prob_cs, 2))
print('-'*80)

# Probability that the customers who have paid back the loan and has met the credit policy
new_df = df[df['paid.back.loan'] == 'Yes']
A_n_B = new_df[new_df['credit.policy'] == 'Yes']
prob_pd_n_cs = len(A_n_B) / len(df)
print("Probability that the customers who have paid back the loan and has met the credit policy is",round(prob_pd_n_cs, 2))
print('-'*80)

# Probability that the customer has paid back the loan given that the credit policy was followed
prob_pd_cs = prob_pd_n_cs / prob_lp
print("Probability that the customer has paid back the loan given that the credit policy was followed is",round(prob_pd_cs, 2))
print('-'*80)

# Probability that the customer has followed the credit policy given that he has paid back the loan
bayes = prob_pd_cs * prob_lp / prob_cs
print("Probability that the customer has followed the credit policy given that he has paid back the Loan :",round(bayes, 2))

# code ends here


# --------------
# code starts here
import seaborn as sns

# Visualization of the initial customer data
plt.figure(figsize=[10,7])
plt.xticks(rotation=45)
sns.countplot(df['purpose'])
plt.show()

# Subsetting the customer data who haven't paid back their loan
df1 = df[df['paid.back.loan'] == 'No']

# Visualization of the customer data who havent paid back
plt.figure(figsize=[10,7])
plt.xticks(rotation=45)
sns.countplot(df1['purpose'])
plt.show()
# code ends here


# --------------
# code starts here
inst_median = df['installment'].median()
inst_mean = df['installment'].mean()

# code starts here
inst_median = df['installment'].median()
inst_mean = df['installment'].mean()
print("Median of the overall monthly installments paid by the borrowers is", round(inst_median, 2))
print('-'*50)
print("Mean of the overall monthly installments paid by the borrowers is", round(inst_mean, 2))
print('-'*50)

# Distribution of monthly installments of the borrowers
plt.figure(figsize=[10,7])
df['installment'].plot.hist()
plt.title('Average monthly installments of the borrowers')
plt.xlabel('Monthly installments')
plt.show()

# Annual income distribution of the borrowers
plt.figure(figsize=[10,7])
df['log.annual.inc'].plot.hist()
plt.title('Average annual income of the borrowers')
plt.xlabel('Annual Income(Lpa)')
plt.show()
# code ends here
# code ends here


