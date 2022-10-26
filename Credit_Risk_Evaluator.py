import numpy as np
print('numpy: %s' % np.__version__)

import pandas as pd
print('pandas: %s' % pd.__version__)

from pathlib import Path

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
print('sklearn: %s' % sklearn.__version__)

# Display all the columns for the dataframes (not-truncated)
pd.set_option('display.max_columns', None)

def check_test_and_train_matching_columns():
    # Display warning if columns do not match
    inner_join = set(train_df.columns) & set(test_df.columns)
    full_join = set(train_df.columns) | set(test_df.columns)
    unmatching_columns = list(full_join - inner_join)

    if (len(unmatching_columns) != 0):
        print("columns count does not match at...")
        return unmatching_columns
    else:
        print("columns match!")

train_df = pd.read_csv(Path(r'C:\Users\gaura\PycharmProjects\2020Q1loans.csv'))
test_df = pd.read_csv(Path(r'C:\Users\gaura\PycharmProjects\2019loans.csv'))
check_test_and_train_matching_columns()

# columns match

print(train_df.describe)
# 15240 rows X 84 columns
print(test_df.describe)
#9494 rows X 84 columns


# Convert categorical data to numeric

# Split the columns into quantitative and categorical
all_columns = set(train_df.columns) | set(test_df.columns)
all_quantitative_columns = set(train_df.describe().columns) | set(test_df.describe().columns)
all_categorical_columns = all_columns - all_quantitative_columns
print(all_categorical_columns)

train_df = pd.get_dummies(train_df, columns=all_categorical_columns)
test_df = pd.get_dummies(test_df, columns=all_categorical_columns)
check_test_and_train_matching_columns()
# columns count doest not match at ['hardship_flag_Y]
# note this mismatch is different from that in the notebook

# test_df has this column but train_df does not
# lets add

train_df['hardship_flag_Y'] = 0
check_test_and_train_matching_columns()

print(train_df.columns.tolist())

# Separate target feature for training data

# we will train the model to be sensitive if the loans are of high risk
# note that the target feature is named differently in the notebook
# using that we get error
# define the target feature correctly

target_feature = "target_high_risk"

# Split the training data
X_train  = train_df.drop(columns=[target_feature])
y_train = train_df[[target_feature]].values.ravel()
#
print(X_train.shape, y_train.shape)
#(15240, 92) (15240,0)

# Split the testing data
X_test  = test_df.drop(columns=[target_feature])
y_test = test_df[[target_feature]].values.ravel()
#
print(X_test.shape, y_test.shape)
#(9494, 92) (9494,0)

# Train the Logistic Regression model on the unscaled data and print the model score
logisticRegr = LogisticRegression(
    solver='lbfgs',
    max_iter=100,
    random_state=0
)
logisticRegr.fit(X_train, y_train)
print("LogisticRegression score: ", logisticRegr.score(X_test, y_test))
#LogisticRegression score:  0.5758373709711396

# Train a Random Forest Classifier model and print the model score
randomForestClass = RandomForestClassifier(random_state=0)
randomForestClass.fit(X_train, y_train)
print("RandomForestClassifier score: ", randomForestClass.score(X_test, y_test))
#RandomForestClassifier score:  0.48462186644196337

# Now trying again with scaling
# Scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Logistic Regression model on the scaled data and print the model score
logisticRegr_scaled = LogisticRegression(
    solver='lbfgs',
    max_iter=100,
    random_state=0
)
logisticRegr_scaled.fit(X_train_scaled, y_train)
print("LogisticRegression scaled score: ", logisticRegr_scaled.score(X_test_scaled, y_test))
# LogisticRegression scaled score:  0.48462186644196337

# Train a Random Forest Classifier model on the scaled data and print the model score
randomForestClass_scaled = RandomForestClassifier(random_state=0)
randomForestClass_scaled.fit(X_train_scaled, y_train)
print("RandomForestClassifier scaled score: ", randomForestClass_scaled.score(X_test_scaled, y_test))
#RandomForestClassifier scaled score:  0.48462186644196337


# although it ran but we got following warning
# C:\Users\gaura\anaconda3\lib\site-packages\sklearn\base.py:493: FutureWarning: The feature names should match those that were passed during fit. Starting version 1.2, an error will be raised.
# Feature names must be in the same order as they were in fit.