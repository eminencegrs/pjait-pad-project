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
