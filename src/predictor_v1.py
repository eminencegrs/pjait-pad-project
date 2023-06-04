import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Set working directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'dataset'))

# Read data from the file.
dataset_path = os.path.join(parent_dir, 'dataset', 'questions.csv')
data_frame = pd.read_csv(dataset_path)
columns = [
    'question_id',
    'title',
    'language',
    'score',
    'is_answered',
    'accepted_answer_id',
    'view_count',
    'answer_count',
    'creation_date',
    'last_edit_date',
    'last_activity_date',
    'closed_date',
    'closed_reason',
    'owner.account_id',
    'owner.reputation',
    'tags'
]

# Clearing data.
data_frame = data_frame[columns]
data_frame['accepted_answer_id'] = data_frame['accepted_answer_id'].fillna('N/A')
data_frame['last_edit_date'] = data_frame['last_edit_date'].fillna('N/A')
data_frame['closed_date'] = data_frame['closed_date'].fillna('N/A')
data_frame['closed_reason'] = data_frame['closed_reason'].fillna('N/A')

# Dropping rows with missing values.
data_frame = data_frame.dropna()

# Renaming columns to make them readable.
data_frame.rename(
    columns={
        'question_id': 'ID',
        'title': 'Title',
        'language': 'Language',
        'score': 'Score',
        'is_answered': 'Is Answered?',
        'accepted_answer_id': 'Answer ID',
        'view_count': 'View Count',
        'answer_count': 'Answer Count',
        'creation_date': 'Creation Date',
        'last_edit_date': 'Edit Date',
        'last_activity_date': 'Last Activity Date',
        'closed_date': 'Closed Date',
        'closed_reason': 'Closed Reason',
        'owner.account_id': 'Owner ID',
        'owner.reputation': 'Owner Reputation',
        'tags': 'Tags'
    },
    inplace=True
)

# Select your features. Note that these need to be numeric for logistic regression.
# Here we select 'Score', 'View Count', 'Answer Count', and 'Owner Reputation'.
feature_columns = ['Score', 'View Count', 'Answer Count', 'Owner Reputation']
X = data_frame[feature_columns]

# Convert the target variable to numeric
y = data_frame['Is Answered?'].astype(int)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=27)

# Logistic Regression with 'liblinear' solver.
classifier_lr = LogisticRegression(solver='liblinear', random_state=27)
classifier_lr.fit(X_train, y_train)
y_pred_lr = classifier_lr.predict(X_test)
print("Logistic Regression with 'liblinear' solver:\n", classification_report(y_test, y_pred_lr))

# Logistic Regression with 'newton-cg' solver.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
classifier_lr_newton = LogisticRegression(solver='newton-cg', random_state=27, max_iter=10000)
classifier_lr_newton.fit(X_train_scaled, y_train)
y_pred_lr_newton = classifier_lr_newton.predict(X_test_scaled)
print("Logistic Regression with 'newton-cg' solver:\n", classification_report(y_test, y_pred_lr_newton))

# Logistic Regression with 'lbfgs' solver.
classifier_lr_lbfgs = LogisticRegression(solver='lbfgs', random_state=27)
classifier_lr_lbfgs.fit(X_train, y_train)
y_pred_lr_lbfgs = classifier_lr_lbfgs.predict(X_test)
print("Logistic Regression with 'lbfgs' solver:\n", classification_report(y_test, y_pred_lr_lbfgs))

# Logistic Regression with 'sag' solver.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
classifier_lr_sag = LogisticRegression(solver='sag', random_state=27, max_iter=10000, C=10)
classifier_lr_sag.fit(X_train_scaled, y_train)
y_pred_lr_sag = classifier_lr_sag.predict(X_test_scaled)
print("Logistic Regression with 'sag' solver:\n", classification_report(y_test, y_pred_lr_sag))

# Logistic Regression with 'saga' solver.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
classifier_lr_sag = LogisticRegression(solver='saga', random_state=27, max_iter=10000, C=10)
classifier_lr_sag.fit(X_train_scaled, y_train)
y_pred_lr_saga = classifier_lr_sag.predict(X_test_scaled)
print("Logistic Regression with 'saga' solver:\n", classification_report(y_test, y_pred_lr_saga))

# Decision Tree
classifier_dt = DecisionTreeClassifier(random_state=27)
classifier_dt.fit(X_train, y_train)
y_pred_dt = classifier_dt.predict(X_test)
print("Decision Tree:\n", classification_report(y_test, y_pred_dt))

# Random Forest
classifier_rf = RandomForestClassifier(random_state=27)
classifier_rf.fit(X_train, y_train)
y_pred_rf = classifier_rf.predict(X_test)
print("Random Forest:\n", classification_report(y_test, y_pred_rf))

# Support Vector Machines
classifier_svm = svm.SVC(random_state=27)
classifier_svm.fit(X_train, y_train)
y_pred_svm = classifier_svm.predict(X_test)
print("Support Vector Machines:\n", classification_report(y_test, y_pred_svm))


### Results:
# The metrics you've provided are precision, recall, F1 score, and accuracy, 
# which are used to measure the performance of a classifier. 
# 
# Here is an interpretation of your results:

# 1. **Logistic Regression with 'liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga' solvers**:
#      All these models have the same performance.
#      The weighted average accuracy is 0.81, which suggests that the models correctly predict
#      the class for 81% of cases in your dataset.
#      The precision for class 0 is high (0.93), meaning when the model predicts class 0,
#      it is correct 93% of the time. However, recall for class 0 is lower (0.73),
#      indicating that the model is able to identify 73% of actual class 0 cases.
#      For class 1, the model has lower precision (0.71) but high recall (0.92/0.93),
#      indicating it can correctly identify a large proportion of actual class 1 cases,
#      but it also misclassifies a number of class 0 cases as class 1.
#      The F1 score, which is a harmonic mean of precision and recall, is 0.82 for class 0 and 0.80 for class 1,
#      showing a balanced performance between precision and recall for both classes.
#
# 2. **Decision Tree**: The Decision Tree model has a slightly lower performance compared
#      to the Logistic Regression models with an accuracy of 0.80.
#      It has lower precision and recall for both classes than the Logistic Regression models.
#      The F1 scores for the Decision Tree model are 0.83 for class 0 and 0.75 for class 1,
#      indicating a slightly imbalanced performance with respect to precision and recall.
#
# 3. **Random Forest**: The Random Forest model has a slightly better accuracy 
#      than the Logistic Regression models and Decision Tree model with a value of 0.82.
#      The precision, recall, and F1 score for class 0 is higher than the Decision Tree model
#      but lower than the Logistic Regression models. However, for class 1,
#      the recall and F1 score is higher than both Logistic Regression and Decision Tree models,
#      and the precision is higher than Logistic Regression but lower than Decision Tree. 
#
# To summarize, the Random Forest model is performing slightly better overall, compared to the other models.
# The Logistic Regression models have consistent results regardless of the solver used,
# and they perform better in identifying class 0 but worse at class 1 compared to the Random Forest model.
# The Decision Tree model has the lowest performance of the three.
#
# These results might vary depending on the specific data distribution of the dataset,
# and it might be beneficial to perform hyperparameter tuning or use other techniques
# such as ensemble methods to improve the models' performances.
# If the classes are imbalanced, using a technique such as SMOTE might be beneficial
# to balance the dataset and improve the performance of the models.
