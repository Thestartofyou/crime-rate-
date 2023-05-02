# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data into a pandas DataFrame
crime_data = pd.read_csv('crime_data.csv')

# Preprocess data by dropping unnecessary columns and filling missing values
crime_data = crime_data.drop(['ID', 'Date', 'Time', 'Location'], axis=1)
crime_data = crime_data.fillna(0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(crime_data.drop('Crime', axis=1), crime_data['Crime'], test_size=0.2, random_state=42)

# Train a logistic regression model on the training data
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train, y_train)

# Test the model on the testing data
y_pred = logreg.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

