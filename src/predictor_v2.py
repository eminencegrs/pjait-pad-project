import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.feature_extraction import FeatureHasher
from datetime import datetime

from column_renamer import ColumnRenamer
from column_filter import ColumnFilter
from data_cleaner import DataCleaner
from data_enhancer import DataEnhancer
from data_reader import DataReader

### Read & prepare data.
# Read data.
data_reader = DataReader('questions.csv')
data_frame = data_reader.read_data()

# Choose the required columns only.
column_filter = ColumnFilter()
data_frame = column_filter.filter_data(data_frame)

# Clear data.
data_cleaner = DataCleaner()
data_frame = data_cleaner.clean_data(data_frame)

# Rename columns to make them readable.
renamer = ColumnRenamer()
data_frame = renamer.rename_columns(data_frame)

# Extend the data frame with additional columns.
enhancer = DataEnhancer()
df = enhancer.enhance(data_frame)

### 

# Convert 'Creation Date' to datetime and then to the timestamp
df['Creation Date'] = pd.to_datetime(df['Creation Date'])
df['date_timestamp'] = df['Creation Date'].map(datetime.toordinal)

# Group by date and language, and count the number of questions asked each day for each language
df_grouped = df.groupby(['date_timestamp', 'Language']).size().reset_index(name='question_count')

# Now split into X and y
X = df_grouped[['date_timestamp', 'Language']]
y = df_grouped['question_count']

# Transform 'Language' into a list of single-item lists
X['Language'] = X['Language'].apply(lambda x: [x])

# Use feature hashing to convert 'Language' into a format that can be used in a regression model
hasher = FeatureHasher(n_features=10, input_type='string')
hashed_features = hasher.transform(X['Language']).toarray()

# Add the hashed features without dropping 'Language'
hashed_features_df = pd.DataFrame(hashed_features, columns=['hash'+str(i) for i in range(hashed_features.shape[1])])
X = pd.concat([X.reset_index(drop=True), hashed_features_df], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the model
regressor = LinearRegression()  
regressor.fit(X_train.drop('Language', axis=1), y_train) 

# Predict the number of questions
y_pred = regressor.predict(X_test.drop('Language', axis=1))

# To compare the actual output values for X_test with the predicted values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Merge back 'Language' and 'date_timestamp'
df['Language'] = X_test['Language'].apply(lambda x: x[0])
df['date_timestamp'] = X_test['date_timestamp']

df['date'] = df['date_timestamp'].apply(lambda x: datetime.fromordinal(x))

# Sort by 'Language' and 'date_timestamp'
df.sort_values(by=['Language', 'date'], inplace=True)

# For each language
languages = df['Language'].unique()
for language in languages:
    df_lang = df[df['Language'] == language]
    plt.figure(figsize=(16,9))
    plt.plot(df_lang['date'], df_lang['Actual'], label='Actual')
    plt.plot(df_lang['date'], df_lang['Predicted'], label='Predicted')

    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.title(f'Actual vs Predicted Over Time for {language}')
    plt.legend()
    plt.show()
