import pandas as pd

from datetime import datetime

from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class QuestionsNumberPredictor:
    def fetch_prediction_results(df):
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)

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
        
        return df
