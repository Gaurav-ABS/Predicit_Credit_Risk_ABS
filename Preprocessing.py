import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

columns = [
    "loan_amnt", "int_rate", "installment", "home_ownership", "annual_inc",
    "verification_status", "pymnt_plan", "dti", "delinq_2yrs",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "total_acc",
    "initial_list_status", "out_prncp", "out_prncp_inv", "total_pymnt",
    "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_amnt", "collections_12_mths_ex_med", "policy_code",
    "application_type", "acc_now_delinq", "tot_coll_amt", "tot_cur_bal",
    "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m",
    "mths_since_rcnt_il", "total_bal_il", "il_util", "open_rv_12m",
    "open_rv_24m", "max_bal_bc", "all_util", "total_rev_hi_lim", "inq_fi",
    "total_cu_tl", "inq_last_12m", "acc_open_past_24mths", "avg_cur_bal",
    "bc_open_to_buy", "bc_util", "chargeoff_within_12_mths", "delinq_amnt",
    "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl", "mort_acc", "mths_since_recent_bc",
    "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats",
    "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75",
    "pub_rec_bankruptcies", "tax_liens", "tot_hi_cred_lim",
    "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit",
    "hardship_flag", "debt_settlement_flag",
    "loan_status"
]

target = "loan_status"


# Load the data
df1 = pd.read_csv(Path('LoanStats_2019Q1.csv.zip'), skiprows=1)[:-2]
df2 = pd.read_csv(Path('LoanStats_2019Q2.csv.zip'), skiprows=1)[:-2]
df3 = pd.read_csv(Path('LoanStats_2019Q3.csv.zip'), skiprows=1)[:-2]
df4 = pd.read_csv(Path('LoanStats_2019Q4.csv.zip'), skiprows=1)[:-2]

df = pd.concat([df1, df2, df3, df4]).loc[:, columns].copy()

# Drop the null columns where all values are null
df = df.dropna(axis='columns', how='all')

# Drop the null rows
df = df.dropna()

# Remove the `Issued` loan status
issued_mask = df['loan_status'] != 'Issued'
df = df.loc[issued_mask]

# convert interest rate to numerical
df['int_rate'] = df['int_rate'].str.replace('%', '')
df['int_rate'] = df['int_rate'].astype('float') / 100


# Convert the target column values to low_risk and high_risk based on their values
x = {'Current': 'low_risk'}
df = df.replace(x)

x = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period'], 'high_risk')
df = df.replace(x)


low_risk_rows = df[df[target] == 'low_risk']
high_risk_rows = df[df[target] == 'high_risk']

#df = pd.concat([low_risk_rows, high_risk_rows.sample(n=len(low_risk_rows), replace=True)])
df = pd.concat([low_risk_rows.sample(n=len(high_risk_rows), random_state=42), high_risk_rows])
df = df.reset_index(drop=True)
df = df.rename({target:'target'}, axis="columns")
df


# save the processed data
df.to_csv('../2019loans.csv', index=False)


# Load the data
validate_df = pd.read_csv(Path('LoanStats_2020Q1.csv.zip'), skiprows=1)[:-2]
validate_df = validate_df.loc[:, columns].copy()

# Drop the null columns where all values are null
validate_df = validate_df.dropna(axis='columns', how='all')

# Drop the null rows
validate_df = validate_df.dropna()

# Remove the `Issued` loan status
issued_mask = validate_df[target] != 'Issued'
validate_df = validate_df.loc[issued_mask]

# convert interest rate to numerical
validate_df['int_rate'] = validate_df['int_rate'].str.replace('%', '')
validate_df['int_rate'] = validate_df['int_rate'].astype('float') / 100


# Convert the target column values to low_risk and high_risk based on their values
x = dict.fromkeys(['Current', 'Fully Paid'], 'low_risk')
validate_df = validate_df.replace(x)

x = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period', 'Charged Off'], 'high_risk')
validate_df = validate_df.replace(x)

low_risk_rows = validate_df[validate_df[target] == 'low_risk']
high_risk_rows = validate_df[validate_df[target] == 'high_risk']

validate_df = pd.concat([low_risk_rows.sample(n=len(high_risk_rows), random_state=37), high_risk_rows])
validate_df = validate_df.reset_index(drop=True)
validate_df = validate_df.rename({target:'target'}, axis="columns")
validate_df

validate_df.to_csv('../2020Q1loans.csv', index=False)

